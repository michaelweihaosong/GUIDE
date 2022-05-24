#%%
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import argparse
import pickle
from utils import *
from models import *
from scipy.linalg import eigh
from scipy.sparse import csgraph
from scipy.sparse.csgraph import laplacian
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.utils import dropout_adj, convert
import warnings
warnings.filterwarnings("ignore")

class PFR:
    def __init__(self, k, W_s, W_F, gamma = 1.0, exp_id='', alpha = None, normed = False):
        """
        Initializes the model.
        :param k:       Hyperparam representing the number of latent dimensions.
        :param W_s:     The adjacency matrix of k-nearest neighbour graph over input space X
        :param W_F:     The adjacency matrix of the pairwise fairness graph G associated to the problem.
        :param nn_k:    Hyperparam that controls the number of neighbours considered in the similarity graph.
        :param gamma:   Hyperparam controlling the influence of W^F.
        :param alpha:   Hyperparam controlling the influence of W^X. If none, default is set to 1 - gamma.
        """
        self.k = k
        self.W_F = W_F
        self.W_s = W_s
        self.exp_id = exp_id
        self.gamma = gamma
        if alpha != None:
            self.alpha = alpha
        else:
            self.alpha = 1 - self.gamma
        self.normed = normed

    def fit(self, X):
        """
        Learn the model using the training data.
        :param X:     Training data.
        """
        print('Just fitting')
        W = (self.alpha * self.W_s) + (self.gamma * self.W_F)
        L, diag_p = csgraph.laplacian(W, normed=self.normed, return_diag=True)

        # - Formulate the eigenproblem.
        lhs_matrix = (X.T.dot(L.dot(X)))
        rhs_matrix = None

        # - Solve the problem
        eigval, eigvec = eigh(a=lhs_matrix,
                                b=rhs_matrix,
                                overwrite_a=True,
                                overwrite_b=True,
                                check_finite=True)
        eigval = np.real(eigval)
        self.k = min(self.k, eigval.shape[0]-1)
        # - Select eigenvectors based on eigenvalues
        # -- get indices of k smallest eigen values
        k_eig_ixs = np.argpartition(eigval, self.k)[:self.k]

        # -- columns of eigvec are the eigen vectors corresponding to the eigen values
        # --- select column vectors corresponding to k largest eigen values
        self.V = eigvec[:, k_eig_ixs]

    def transform(self, X):
        return (self.V.T.dot(X.T)).T

    def project(self, X):
        Z = self.transform(X)
        proj = Z.dot(self.V.T)
        np.save('AFR_k{}_p{}_id{}_'.format(self.k, self.nn_k, self.exp_id), proj)
        return proj

    def fit_transform(self, X):
        """
        Learns the model from the training data and returns the data in the new space.
        :param X:   Training data.
        :return:    Training data in the new space.
        """
        print('Fitting and transforming...')
        self.fit(X)
        return self.transform(X)



# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
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
parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument('--gamma', type=float, default=0.5,
                    help='tradeoff coefficient for the two matrices')

parser.add_argument('--dataset', type=str, default='credit',
                    choices=['credit', 'pokec_n', 'income'])
parser.add_argument('--model', type=str, default='gcn',
                    choices=['gcn', 'gin', 'jk'])

args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()


# set seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(f"Working on PFR_gnn {args.dataset}, model {args.model}, gamma {args.gamma}")
# torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





# Load data
# print(args.dataset)

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

    
edge_index = convert.from_scipy_sparse_matrix(adj)[0]
sim = calculate_similarity_matrix(adj, features, metric='cosine')
lap = laplacian(sim)

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


# preprocess features using PFR
print(f"Fitting PFR preprocessing")
PFR_model = PFR(16, adj, sim, gamma = args.gamma)
features = PFR_model.fit_transform(features.detach().cpu().numpy())
print(f"Embedding shape: {features.shape}")
features = torch.FloatTensor(features)


    
num_class = 1
if args.model == 'gcn':
	model = GCN(nfeat=features.shape[1],
	            nhid=args.hidden,
	            nclass=num_class,
	            dropout=args.dropout)
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	model = model.to(device)


elif args.model == 'gin':
	model = GIN(nfeat=features.shape[1],
	            nhid=args.hidden,
	            nclass=num_class,
	            dropout=args.dropout)
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	model = model.to(device)

elif args.model == 'jk':
	model = JK(nfeat=features.shape[1],
	            nhid=args.hidden,
	            nclass=num_class,
	            dropout=args.dropout)
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	model = model.to(device)

    
    

# Train model
t_total = time.time()
best_loss = np.inf
best_acc = 0
features = features.to(device)
edge_index = edge_index.to(device)
labels = labels.to(device)
lap = lap.to(device)
lap_1 = lap_1.to(device)
lap_2 = lap_2.to(device)


for epoch in range(args.epochs+1):
    t = time.time()

    if args.model in ['gcn', 'gin', 'jk']:
        model.train()
        optimizer.zero_grad()
        output = model(features, edge_index)

        # Binary Cross-Entropy  
        preds = (output.squeeze()>0).type_as(labels)
        loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float().to(device))
        
        auc_roc_train = roc_auc_score(labels.cpu().numpy()[idx_train], output.detach().cpu().numpy()[idx_train])
        loss_train.backward()
        optimizer.step()

        # Evaluate validation set performance separately,
        model.eval()
        output = model(features, edge_index)

        # Binary Cross-Entropy
        preds = (output.squeeze()>0).type_as(labels)
        loss_val = F.binary_cross_entropy_with_logits(output[idx_val], labels[idx_val].unsqueeze(1).float().to(device))
        

        auc_roc_val = roc_auc_score(labels.cpu().numpy()[idx_val], output.detach().cpu().numpy()[idx_val])

        # if epoch % 100 == 0:
        #     print(f"[Train] Epoch {epoch}:train_loss: {loss_train.item():.4f} | train_auc_roc: {auc_roc_train:.4f} | val_loss: {loss_val.item():.4f} | val_auc_roc: {auc_roc_val:.4f}")

        if loss_val.item() < best_loss:
            best_loss = loss_val.item()
            weightName = './torch_weights/PFR_gnn_weights.pt'
            torch.save(model.state_dict(), weightName)


# print("Optimization Finished!")
# print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

if args.model in ['gcn', 'gin', 'jk']:
    model.load_state_dict(torch.load(weightName))
    model.eval()
    output = model(features.to(device), edge_index.to(device))
    counter_features = features.clone()
    counter_features[:, sens_idx] = 1 - counter_features[:, sens_idx]
    counter_output = model(counter_features.to(device), edge_index.to(device))
    noisy_features = features.clone() + torch.ones(features.shape).normal_(0, 1).to(device)
    noisy_output = model(noisy_features.to(device), edge_index.to(device))



# Report
output_preds = (output.squeeze()>0).type_as(labels)
counter_output_preds = (counter_output.squeeze()>0).type_as(labels)
noisy_output_preds = (noisy_output.squeeze()>0).type_as(labels)
auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu()], output.detach().cpu().numpy()[idx_test.cpu()])
counterfactual_fairness = 1 - (output_preds.eq(counter_output_preds)[idx_test].sum().item()/idx_test.shape[0])
robustness_score = 1 - (output_preds.eq(noisy_output_preds)[idx_test].sum().item()/idx_test.shape[0])

individual_unfairness = torch.trace( torch.mm( output.t(), torch.sparse.mm(lap, output) ) ).item()
f_u1 = torch.trace(torch.mm(output.t(), torch.sparse.mm(lap_1, output)))/m_u1
f_u1 = f_u1.item()
f_u2 = torch.trace(torch.mm(output.t(), torch.sparse.mm(lap_2, output)))/m_u2
f_u2 = f_u2.item()
if_group_pct_diff = np.abs(f_u1-f_u2)/min(f_u1, f_u2)
GDIF = max(f_u2/f_u1, f_u1/f_u2)

# print report
print("The AUCROC of estimator: {:.4f}".format(auc_roc_test))
print(f'Total Individual Unfairness: {individual_unfairness}')
print(f'Individual Unfairness for Group 1: {f_u1}')
print(f'Individual Unfairness for Group 2: {f_u2}')
print(f'GDIF: {GDIF}')
