## GUIDE: Group Equality Informed Individual Fairness in Graph Neural Networks



## 1. Setup

### Installing software
Please run the following commands to install necessary packages.
For more details on Pytorch Geometric please refer to install the PyTorch Geometric packages following the instructions from [here.](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)



```
conda create --name guide python==3.7.11
conda install pytorch==1.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge

pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
pip install torch-geometric==2.0.1

pip install aif360==0.3.0
```


## 2. Datasets
We ran our experiments on two high-stakes datasets: credit and income and also a larger social network dataset pokec. All the data are present in the './dataset' folder. Due to space constraints, pokec is zipped so please unzip it before use. 

## 3. Usage
The main scripts are listed below: GUIDE, InFoRM, PFR, FairGNN, NIFTY, and vanilla are included.

### Examples
run GUIDE: Evaluate fairness and utility performance of GCN and credit dataset

`python run_guide.py --model gcn --dataset credit --alpha 5e-6 --beta 1 --seed 1`
<p align="left"><i>
  The AUCROC of estimator: 0.6776<br/>
  Total Individual Unfairness: 1886.4669<br/>
  GDIF: 1.0030<br/>
</i></p>

run InFoRM: Evaluate fairness and utility performance of InFoRM-GCN and credit dataset

`python run_inform.py --model gcn --dataset credit --alpha 5e-7 --opt_if 1 --seed 1`
<p align="left"><i>
  The AUCROC of estimator: 0.6813<br/>
  Total Individual Unfairness: 2408.4109<br/>
  GDIF: 1.4903<br/>
</i></p>  

run PFR: Evaluate fairness and utility performance of PFR-GCN and credit dataset

`python run_PFR_gnn.py --model gcn --dataset credit --gamma 0.5 --seed 1`
<p align="left"><i>
  The AUCROC of estimator: 0.6724<br/>
  Total Individual Unfairness: 12494.9785<br/>
  GDIF: 1.3037<br/>
</i></p>   

run FairGNN: Evaluate fairness and utility performance of FairGNN-GCN and credit dataset

`python baseline_fairGNN.py --model gcn --dataset credit --alpha 4 --beta 1000 --seed 1`
<p align="left"><i>
  The AUCROC of estimator: 0.6890<br/>
  Total Individual Unfairness: 43560.7930<br/>
  GDIF: 1.3326<br/>
</i></p>  

run NIFTY: Evaluate fairness and utility performance of NIFTY-GCN and credit dataset

`python run_nifty.py --encoder gcn --dataset credit --model ssf --sim_coeff 0.5 --seed 1`
<p align="left"><i>
  The AUCROC of estimator: 0.6927<br/>
  Total Individual Unfairness: 31671.1328<br/>
  GDIF: 1.2466<br/>
</i></p>  

run vanilla GNN: Evaluate fairness and utility performance of GCN and credit dataset (note it is using run_inform.py but with opt_if flag off)

`python run_inform.py --model gcn --dataset credit --alpha 0 --opt_if 0 --seed 1`
<p align="left"><i>
  The AUCROC of estimator: 0.6984<br/>
  Total Individual Unfairness: 40829.4336<br/>
  GDIF: 1.3748<br/>
</i></p>  

## 4. Licenses
Note that the code in this repository is licensed under MIT License. Please carefully check them before use. 

