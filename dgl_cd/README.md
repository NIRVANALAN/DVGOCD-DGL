# experiments

- Optimizer: Adam
- L2 loss + BP loss

| Model | hidden_number | weight_decay | Final NMI | 
| --- | --- | --- | ---  | 
| GCN | 8 | 1e-5 | 37.3 - 48 |
| GCN | 8 | 5e-3 | 46.6 | 
| GCN | **64** | 5e-3 | 47.1, 45.85 | 
| GAT | 8*8 | 5e-4 | 44.97 | 

## Questions
1. Dropout 
2. weight_decay no bias?
3. Unstable training / convergence.
