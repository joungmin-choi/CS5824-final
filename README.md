# Breast Cancer Subtype Classification Based on Optimized BP Neural Network

In this project, we presented optimized backpropagation neural network models for breast cancer subtype classification based on six optimization algorithms utilizing gene expression profiles. The optimization algorithms were investigated for their potential to improve the performance of breast cancer subtype prediction tasks. They were evaluated and compared to the widely-used Glorot and random initialization in the aspect of f1-score and accuracy. 

## Requirements
* Keras (>= 2.7.0)
* Python (>= 3.5)
* Python packages : numpy, pandas, os, sys
* R (>= 4.2.0)
* R package : Deseq2

## Usage
1. Feature selection
**"BRCA_gene_expression_tumor_normal.csv"** file is a RNA-seq dataset with a dataframe contatining gene expression of features, where each row and column represent **gene** and **sample**, respectively. Example for dataset format is provided below.

```
feature,TCGA-AR-A2LE,TCGA-BH-A0B7,TCGA-E2-A1IJ,...
A1BG,0,10,30,
A1CF,20,40,0
A2M,60,25,0
...
```
In **"feature-selction"** directory, **"coldata.csv"** file is a dataframe-based dataset, where each row represents **sample**, and column represents **type** (e.g. tumor, normal) of each sample. Example for dataset format is provided below. below.

```
sample,type
TCGA-AR-A2LE,tumor
TCGA-BH-A0B7,normal
TCGA-E2-A1IJ,tumor
...
```

After preparing those datasets, run **feature_selection.R**, which would provide the list of differential expressed genes (DEG) and the normalized reads. 

2. After selecting the DEGs, prepare the final dataset for breast cancer subtype classification based on the BP-optimized neural network, where each row represents **sample**, and column represents **gene**. You can find the example case in the **10cv_datasets** directory, which consists of 10cross-validation datasets used in our project. 

3. Based on the optimization algorithm you want to run, select in the **"BP-optimized-BRCA-subtype-classifiers"** directory and run **"python3 algorithmName-net.py"**.

* Ant lion optimization (ALO): ALO-net.py
* Sine cosine algorithm (SCA): SCA-net.py
* Whale optimization algorithm (WOA): WOA-net.py
* Firefly algorithm (FA): FA-net.py
* Moth-flame algorithm (MFO): MFO-net.py
* Grey Wolf Optimization (GWO): GWO-net.py
* Glorot (Xavier) initialization: Glorot-net.py
* Random initialization: Random-net.py

4. To run machine learning-based classifiers, run **"./run_ml.sh"** in the **"ML_classifiers"** directory. 
