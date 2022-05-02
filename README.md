# ATMOL

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
python             3.7.2
pytorch            1.8.2+cu111
torch-cluster      1.5.9 
torch-geometric    1.7.2      
torch-scatter      2.0.8                    
torch-sparse       0.6.9
torchvision        0.9.2+cu111         
tornado            5.1         
tqdm               4.26.0
pandas             0.23.4 
matplotlib         2.2.2
numpy              1.18.5
scikit-learn       0.22
rdkit              2018.09.1   conda-forge
deepchem           2.6.0.dev20211018190358
```

## Pretrain Dataset
### Dataset
`in-vitro.csv`  includes 306,347 SMILES from ZINC substance channel.

`now.csv`  selected 3,000,000 molecules from the now set(now set includes 9,814,569 SMILES from ZINC substance channel)
### Process
`utils_gat_pretrain.py` make the graph ready from SMILES by PyTorch Geometrics and RDKIT, finally torch.save() to XXX.pt

## Downstream Dataset
### Dataset
For downstream performance evaluation, we chose 7 datasets from MoleculeNet, which collected more than forty molecular property prediction tasks. 

The 7 datasets were BBBP, BACE, HIV, ClinTox, Tox21, SIDER and MUV.

### Process
`utils_clr_downstream.py` make the graph ready from SMILES by PyTorch Geometrics and Deepchem, finally torch.save() to XXX.pt

## ATMOl Usage
### Train Model
```
python main_pretain.py --batch_size 512 --epochs 500 --datafile in-vitro
optional arguments:
--feature_dim                 Feature dim for latent vector [default value is 128]
--temperature                 Temperature used in softmax [default value is 0.5]
--k                           Top k most similar images used to predict the label [default value is 200]
--batch_size                  Number of images in each mini-batch [default value is 512]
--epochs                      Number of sweeps over the dataset to train [default value is 500]
--datafile                    orginal data for smile input [in-vitro, now]
```
### Attention-wise mask for graph augmentation
```
model_gat_pre.py    change mask methods in def attention_del
del_indices = get_allIndex(weight, len(weight)) + count
    chose max-weight or min-weight 
    def get_allIndex():
        # max-weight descending=True  or  min-weight descending=False
        sorted, indices = torch.sort(weight, dim=0, descending=True)

del_indices = get_randomIndex(weight,len(weight))+count
del_indices = get_rouletteIndex(weight, len(weight)) + count
```

### Downstream Evaluation
```
python model_clr_downstream.py --batch_size 128 --epochs 500 
optional arguments:
--model_path                  The pretrained model path [default value is 'results/128_0.5_200_512_500_model.pth']
--batch_size                  Number of images in each mini-batch [default value is 128]
--epochs                      Number of sweeps over the dataset to train [default value is 500]
```

## Results
the model was trained on Three NVIDIA GPU 3090.


