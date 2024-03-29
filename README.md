# Semi-supervised Adaptive Discriminative Discretization (SADD) for Naive Bayes classifier
`SADD` is a semi-supervised discretization framework that can be universally applied in various naive Bayes classifiers. 
![](https://github.com/shellpower96/Semi-supervised-Adaptive-Discriminative-Discretization-for-Naive-Bayes-classifier/blob/main/framework.png)
There are two main stages in `SADD`:
- `Pseudo labeling`

  The k-nearst-neighboors is used to derive the pseudo labels for unlabeled data. The default k is set as 1.
* `Adaptive dicriminative discretization`

  An adaptive threshold is designed to derive the discretization scheme with less information loss.

## Implementation
This is an source code of `SADD` for naive Bayes classifier by using MATLAB. The version of MATLAB should be `>=2019b`.
- To successfully run the code, you need install the `Bioinformatics Toolbox` and `Statistics and Machine Learning Toolbox`.
* Dowload the source file, place it MATLAB and then run the `main.m` file.

## Dataset
We provide a zip file `IndoorLoc.zip` for one of the used datasets in the paper and it contains a CSV file `IndoorLoc.csv` and `read.me` for dataset description.
The example dataset `IndoorLoc.csv` have total 21048 samples, 520 features with 3 classes. The detailed descriptions of all datasets used in the paper can be found in https://archive.ics.uci.edu/ml/index.php.

All the features should be first convert to numerical value, and then make a classification by NB classifiers.

If you use this code, please cite:
```
@article{wang2023semi,
  title={A semi-supervised adaptive discriminative discretization method improving discrimination power of regularized naive Bayes},
  author={Wang, Shihe and Ren, Jianfeng and Bai, Ruibin},
  journal={Expert Systems with Applications},
  volume={225},
  pages={120094},
  year={2023},
  publisher={Elsevier}
}
```
