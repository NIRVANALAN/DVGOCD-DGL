# Overlapping Community Detection with Graph Neural Networks

Pytorch implementation of the **Deep Variational Generative Overlapping Community Detection**

## Usage
The main algorithm and other utilities(metrics, evaluation .etc) are implemented in the `embedding_cd` package that can be installed as
```bash
python setup.py install
```
The main analysis and visualization code are in Jupyter notebooks

run 
```pip install -r requirements.txt``` 
first.

## Requirements
```
numpy>=1.16.4
pytorch>=1.2.0
scipy>=1.3.1
dgl>=0.42
...
```

Some modifications in DGL version:
1. append ```F.relu``` after logits
2. ```BatchNorm``` after GCNLayey
3. ```nocd.utils.l2_reg_loss(model, scale=args.weight_decay)``` rather than setting weight_decay in ```Adam```
4. ```dropout -> Relu -> BN```  
