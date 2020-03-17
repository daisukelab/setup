# setup

Setup scripts/resources for various purposes/projects.

## Profiling

`profiling` folder consists profiling scripts.

### PyTorch Profiling: profile_pytorch_mnist.py

Measures performance for all training processs. Example result :

| index   |   download |   dataloader |   training |   inference x 3 |
|:--------|-----------:|-------------:|-----------:|----------------:|
| 0       |   5.26804  |     2.71466  |  39.6141   |       1.06173   |
| 1       |   5.26432  |     1.56748  |  39.8806   |       1.08058   |
| 2       |   6.19575  |     1.61389  |  39.9198   |       1.0839    |
| mean    |   5.57604  |     1.96534  |  39.8048   |       1.0754    |
| min     |   5.26432  |     1.56748  |  39.6141   |       1.06173   |
| max     |   6.19575  |     2.71466  |  39.9198   |       1.0839    |
| std     |   0.536689 |     0.649341 |   0.166364 |       0.0119574 |

