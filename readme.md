# Learning Domain-Invariant Relationship with Instrumental Variable for Domain Adaptation and Generalization
by xxxx

## Introduction
This repository is for our paper 'xxxxx'  
This paper proposes to estimate the domain-invariant relationship contained in conditional distributions for DA and DG tasks. Specifically, with a causal view on the data generating process, we find that the input of one domain is a valid instrumental variable for other domains. It can theoretically help us consistently estimate the domain-invariant relationship between input and label. Inspired by this finding, we design a unified framework for DA and DG tasks by learning the Domain-invariant Relationship with Instrumental VariablE (DRIVE), which is a simple yet effective method that only adds one fully-connected layer to the common DA/DG framework. We show theoretically and experimentally that DRIVE can learn domain-invariant relationship. Numerous experiments on a variety of real-world datasets demonstrate that DRIVE consistently outperforms other state-of-the-art methods on both DA and DG tasks.

##
-   python 3.6.8 
   
   ```
   conda create -n DRIVE python=3.6.8
   ```
   
-   PyTorch 1.7.0 
    
    ``` bash
    conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
    ```
    
-   Other packages in requirements.txt

## Usage
1. Clone the repository and download the datasets
2. Train the model.