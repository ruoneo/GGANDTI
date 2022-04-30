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
preprocessed_data: Process raw data using matrix completion.  
preprocessed_features: Make similarities data to become sparse, then use matrix completion to process the original features. 
get_latent_variable: The initial features obtained by matrix completion are encoded into suitable low-dimensional features.
main: The model body is stored here. Execute from train.py. 
addition: Additional file, including calculating the mean of ten cross-validations.
```
### requirements:
    pip install -r requirements.txt

### acknowledgements:
Thanks to Yamanish et al. and Luo at al. for the datasets, 
which helped the research to proceed smoothly.

### datasets：
Gold standard datasets from Yamanishi et al. article
(https://doi.org/10.1093/bioinformatics/btn162) and Luo datasets from Luo et al. article
(https://doi.org/10.1038/s41467-017-00680-8).
