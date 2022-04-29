Predicting drug-target interactions based on improved graph generative adversarial network and variational graph auto-encoder
==============
A method for drug-target interactions prediction.  

### usage:
```
1. release the dataset to the src sibling directory
2. open run.py and setup hyperparameters
3. python src/run.py 
```

### folder description:
```
preprocessed_data: 使用矩阵补全处理原始的数据。   
preprocessed_features: 稀疏化相似性数据，然后使用矩阵补全处理原始的特征。  
get_latent_variable: 将矩阵补全得到的初始特征编码成合适的低维特征。  
main: 模型主体存放于此。从train.py开始执行。  
addition: 附加文件，包括计算十次交叉验证的平均值。
```
### requirements:
    pip install requirements.txt

### acknowledgements:
Thanks to Yamanish et al. and Luo at al. for the datasets, 
which helped the research to proceed smoothly.

### datasets：
Gold standard datasets from Yamanishi et al. article
(https://doi.org/10.1093/bioinformatics/btn162) and Luo datasets from Luo et al. article
(https://doi.org/10.1038/s41467-017-00680-8).
