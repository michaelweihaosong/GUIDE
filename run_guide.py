#%%
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from datetime import datetime
import argparse
import pickle
from utils import *
from models import *
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.utils import dropout_adj, convert
from scipy.sparse.csgraph import laplacian



# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')

parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=3000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--heads', type=int, default=1,
                    help='Number of attention heads')
parser.add_argument('--concat', type=bool, default=False,
                    help='whether use concatenation of multi-head attention')                    
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--negative_slope', type=float, default=0.2,
                    help='Negative slope for leaky relu.')
parser.add_argument('--alpha', type=float, default=0.0,
                    help='regularization coeff for the individual fairness objective')
parser.add_argument('--beta', type=float, default=0.0,
                    help='regularization coeff for the GDIF objective')
parser.add_argument('--dataset', type=str, default='credit',
                    choices=['credit', 'pokec_n', 'income'])
parser.add_argument("--num_layers", type=int, default=1,
                        help="number of hidden layers")
parser.add_argument('--model', type=str, default='gcn',
                    choices=['gcn', 'gin', 'jk'])
parser.add_argument('--initialize_training_epochs', type=int, default=1000,
                    help="number of epochs for backbone GNN")



args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()


# # set seeds

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
   torch.cuda.manual_seed(args.seed)

print(args)

# torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# set device
if not args.no_cuda:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

modelWeightsFolder = './torch_weights/'
initEncoderWeightsName = f"{modelWeightsFolder}/guide_initEncoder_weights.pt"
guideEncoderWeightsName = f"{modelWeightsFolder}/guide_guideEncoder_weights.pt"

# Load data

# Load credit_scoring dataset
if args.dataset == 'credit':
	sens_attr = "Age"  # column number after feature process is 1
	sens_idx = 1
	predict_attr = 'NoDefaultNextMonth'
	label_number = 6000
	path_credit = "./dataset/credit"
	adj, features, labels, idx_train, idx_val, idx_test, sens = load_credit(args.dataset, sens_attr,
	                                                                        predict_attr, path=path_credit,
	                                                                        label_number=label_number
	                                                                        )
	norm_features = feature_norm(features)
	norm_features[:, sens_idx] = features[:, sens_idx]
	features = norm_features

elif args.dataset == 'income':
	sens_attr = "race"  # column number after feature process is 1
	sens_idx = 8
	predict_attr = 'income'
	label_number = 3000
	path_income = "./dataset/income"
	adj, features, labels, idx_train, idx_val, idx_test, sens = load_income(args.dataset, sens_attr,
	                                                                        predict_attr, path=path_income,
	                                                                        label_number=label_number
	                                                                        )
	norm_features = feature_norm(features)
	norm_features[:, sens_idx] = features[:, sens_idx]
	features = norm_features
    

elif args.dataset.split('_')[0] == 'pokec':
    if args.dataset == 'pokec_z':
        args.dataset = 'region_job'
    elif args.dataset == 'pokec_n':
        args.dataset = 'region_job_2'
    sens_attr = "AGE"
    predict_attr = "I_am_working_in_field"
    label_number = 10000
    sens_idx = 4
    path="./dataset/pokec/"
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_pokec(args.dataset,sens_attr,
                                                                                    predict_attr,
                                                                                    path=path,
                                                                                    label_number=label_number)


else:
	print('Invalid dataset name!!')
	exit(0)

print(f"Getting similarity matrix...")
edge_index = convert.from_scipy_sparse_matrix(adj)[0]
sim = calculate_similarity_matrix(adj, features, metric='cosine')
sim_edge_index, sim_edge_weight = convert.from_scipy_sparse_matrix(sim)
sim_edge_weight = sim_edge_weight.type(torch.FloatTensor)
lap = laplacian(sim)
print(f"Similarity matrix nonzero entries: {torch.count_nonzero(sim_edge_weight)}")

print("Get laplacians for IFG calculations...")
try:
    with open('./stored_laplacians/' + args.dataset + '.pickle', 'rb') as f:
        loadLaplacians = pickle.load(f)
    lap_list, m_list, avgSimD_list = loadLaplacians['lap_list'], loadLaplacians['m_list'], loadLaplacians['avgSimD_list']
    print("Laplacians loaded from previous runs")
except FileNotFoundError:
    print("Calculating laplacians...(this may take a while for pokec_n)")
    lap_list, m_list, avgSimD_list = calculate_group_lap(sim, sens)
    saveLaplacians = {}
    saveLaplacians['lap_list'] = lap_list
    saveLaplacians['m_list'] = m_list
    saveLaplacians['avgSimD_list'] = avgSimD_list
    with open('./stored_laplacians/' + args.dataset + '.pickle', 'wb') as f:
        pickle.dump(saveLaplacians, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Laplacians calculated and stored.")

        
        
lap = convert_sparse_matrix_to_sparse_tensor(lap)
lap_list = [convert_sparse_matrix_to_sparse_tensor(X) for X in lap_list]
lap_1 = lap_list[0]
lap_2 = lap_list[1]
m_u1 = m_list[0]
m_u2 = m_list[1]

#%%    
# Model and optimizer
#num_class = labels.unique().shape[0]-1
num_class = 1
if args.model == 'gcn':
	model = GCN(nfeat=features.shape[1],
	            nhid=args.hidden,
	            nclass=num_class,
	            dropout=args.dropout)
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


elif args.model == 'gin':
	model = GIN(nfeat=features.shape[1],
	            nhid=args.hidden,
	            nclass=num_class,
	            dropout=args.dropout)
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


elif args.model == 'jk':
	model = JK(nfeat=features.shape[1],
	            nhid=args.hidden,
	            nclass=num_class,
	            dropout=args.dropout)
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)



model = model.to(device)

ifgModel = guideEncoder(num_layers = args.num_layers, nfeat=args.hidden,
            nhid=args.hidden,
            nclass=num_class,
            heads=args.heads,
            negative_slope=args.negative_slope,
            concat=args.concat,                     
            dropout=args.dropout)
ifgOptimizer = optim.Adam(ifgModel.parameters(), lr=args.lr, weight_decay=args.weight_decay)

ifgModel = ifgModel.to(device)



# Train model
t_total = time.time()
best_perf_val = 0
best_total_loss_val = np.inf
features = features.to(device)
edge_index = edge_index.to(device)
labels = labels.to(device)
sim_edge_index = sim_edge_index.to(device)
sim_edge_weight = sim_edge_weight.to(device)
lap = lap.to(device)
lap_1 = lap_1.to(device)
lap_2 = lap_2.to(device)

print(f"---------------Embedding Initialization---------------------")
################################Initialize Embedding################################
for epoch in range(args.initialize_training_epochs+1):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, edge_index)

    loss_label_init_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float().to(device))

    auc_roc_init_train = roc_auc_score(labels.cpu().numpy()[idx_train], output.detach().cpu().numpy()[idx_train])
    individual_unfairness_vanilla = torch.trace(torch.mm(output.t(), torch.sparse.mm(lap,output))).item()
    f_u1 = torch.trace(torch.mm(output.t(), torch.sparse.mm(lap_1, output)))/m_u1
    f_u1 = f_u1.item()
    f_u2 = torch.trace(torch.mm(output.t(), torch.sparse.mm(lap_2, output)))/m_u2
    f_u2 = f_u2.item()
    GDIF_vanilla = max(f_u1/f_u2, f_u2/f_u1)

    loss_label_init_train.backward()
    optimizer.step()
    ################################Logging################################
    if epoch % 100 == 0:
        print(f"----------------------------")
        print(f"[Train] Epoch {epoch}: ")
        print(f"---Embedding Initialize---")
        print(f"Embedding Initialize: loss_label_train: {loss_label_init_train.item():.4f}, auc_roc_train: {auc_roc_init_train:.4f}, individual_unfairness_vanilla: {individual_unfairness_vanilla:.4f}, GDIF_vanilla {GDIF_vanilla:.4f}")



        
print(f"--------------------Training GUIDE--------------------------")
for epoch in range(args.epochs+1):
    t = time.time()
    
    ################################Training################################
    
    ifgModel.train()
    ifgOptimizer.zero_grad()
    with torch.no_grad():
        output = model.body(features, edge_index)
    ifgOutput = ifgModel(output, sim_edge_index, sim_edge_weight)

    loss_label_guide_train = F.binary_cross_entropy_with_logits(ifgOutput[idx_train], labels[idx_train].unsqueeze(1).float().to(device))
    auc_roc_guide_train = roc_auc_score(labels.cpu().numpy()[idx_train], ifgOutput.detach().cpu().numpy()[idx_train])
    ifair_loss = torch.trace( torch.mm( ifgOutput.t(), torch.sparse.mm(lap, ifgOutput) ) )
    f_u1 = torch.trace(torch.mm(ifgOutput.t(), torch.sparse.mm(lap_1, ifgOutput)))/m_u1
    f_u2 = torch.trace(torch.mm(ifgOutput.t(), torch.sparse.mm(lap_2, ifgOutput)))/m_u2
    GDIF = max(f_u1/f_u2, f_u2/f_u1)
    ifg_loss = (f_u1/f_u2-1)**2 + (f_u2/f_u1-1)**2

    

    loss_guide_train = loss_label_guide_train + args.alpha * ifair_loss + args.beta * ifg_loss
    loss_guide_train.backward()
    ifgOptimizer.step()

    ################################Validation################################
    # Evaluate validation set performance separately
    ifgModel.eval()
    ifgOutput = ifgModel(output, sim_edge_index, sim_edge_weight)


    # Get validation losses for guide encoder    
    # Label loss
    loss_label_guide_val = F.binary_cross_entropy_with_logits(ifgOutput[idx_val], labels[idx_val].unsqueeze(1).float().to(device))
    # Individual unfairness loss
    individual_unfairness = torch.trace(torch.mm(ifgOutput.t(), torch.sparse.mm(lap, ifgOutput))).item()
    # IF Group loss
    f_u1 = torch.trace(torch.mm(ifgOutput.t(), torch.sparse.mm(lap_1, ifgOutput)))/m_u1
    f_u1 = f_u1.item()
    f_u2 = torch.trace(torch.mm(ifgOutput.t(), torch.sparse.mm(lap_2, ifgOutput)))/m_u2
    f_u2 = f_u2.item()
    GDIF = max(f_u1/f_u2, f_u2/f_u1)
    ifg_loss = (f_u1/f_u2-1)**2 + (f_u2/f_u1-1)**2

    # Reporting metrics
    preds_guide = (ifgOutput.squeeze()>0).type_as(labels)
    auc_roc_guide_val = roc_auc_score(labels.cpu().numpy()[idx_val], ifgOutput.detach().cpu().numpy()[idx_val])
    if_reduction = (individual_unfairness_vanilla-individual_unfairness)/individual_unfairness_vanilla

    perf_val = auc_roc_guide_val + if_reduction + (GDIF_vanilla-GDIF)/(GDIF_vanilla-1)
    total_loss_val = loss_label_guide_val + args.alpha * individual_unfairness + args.beta * ifg_loss
    if (total_loss_val < best_total_loss_val) and (epoch > 500):
        best_total_loss_val = total_loss_val
        torch.save(ifgModel.state_dict(), guideEncoderWeightsName)
    ################################Logging################################
    if epoch % 100 == 0:
        print(f"----------------------------")
        print(f"[Train] Epoch {epoch}: ")
        #print(f"output {output}")
        #print(f"ifgOutput {ifgOutput}")
        print(f"---Training All objectives---")
        print(f"loss_label train {loss_label_guide_train.item():.4f}, auc_roc_train: {auc_roc_guide_train.item():.4f}")
        print(f"---Validation---")
        print(f"individual_unfairness_vanilla {individual_unfairness_vanilla:.4f}, GDIF_vanilla {GDIF_vanilla:.4f}")
        print(f"loss_total_val: {total_loss_val:.4f}, loss_label_val: {loss_label_guide_val.item():.4f}, loss_ifair: {args.alpha * individual_unfairness:.4f}, loss_ifg: {args.beta*ifg_loss:.4f}, auc_roc_val: {auc_roc_guide_val:.4f}, Individual Fairness: {individual_unfairness:.4f}, if_reduction: {'{:.2%}'.format(if_reduction)}, GDIF: {GDIF:.4f}, Perf_val: {perf_val:.4f}")


torch.save(model.state_dict(), initEncoderWeightsName)




################################Testing################################


model.load_state_dict(torch.load(initEncoderWeightsName))
model.eval()
output = model.body(features, edge_index)

ifgModel.load_state_dict(torch.load(guideEncoderWeightsName))
ifgModel.eval()
output, attention_weights = ifgModel( output, sim_edge_index.to(device), sim_edge_weight.to(device), return_attention_weights=True )

attention_weights = torch.sparse_coo_tensor(attention_weights[0], attention_weights[1])
attention_weights = attention_weights.detach()
attention_weights = attention_weights.coalesce()


# Report
output_preds = (output.squeeze()>0).type_as(labels)
auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu()], output.detach().cpu().numpy()[idx_test.cpu()])

individual_unfairness = torch.trace( torch.mm( output.t(), torch.sparse.mm(lap, output) ) ).item()
f_u1 = torch.trace(torch.mm(output.t(), torch.sparse.mm(lap_1, output)))/m_u1
f_u1 = f_u1.item()
f_u2 = torch.trace(torch.mm(output.t(), torch.sparse.mm(lap_2, output)))/m_u2
f_u2 = f_u2.item()
if_group_pct_diff = np.abs(f_u1-f_u2)/min(f_u1, f_u2)
GDIF = max(f_u1/f_u2, f_u2/f_u1)

# print report
print("---Testing---")

print("The AUCROC of estimator: {:.4f}".format(auc_roc_test))
print(f'Total Individual Unfairness: {individual_unfairness}')
print(f'Individual Unfairness for Group 1: {f_u1}')
print(f'Individual Unfairness for Group 2: {f_u2}')
print(f'GDIF: {GDIF}')
   
