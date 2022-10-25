Predicting drug-target interactions based on improved graph generative adversarial network and variational graph auto-encoder
==============
A framework for drug-target interactions prediction.

### Usage:

1. Release the dataset to the `src` sibling directory `data/RawData`
2. Open run.py and setup hyper-parameters in `src/config.py`
3. Use the following commands to perform training and validation.

```cmd
python run.py
```

4. (Optional) Repeat steps 2 and 3 10 times to get the results of 10 times ten-fold CVs. The results are stored in results/{dataset}/final_results. Finally, execute the following command to calculate the average.

```cmd
python calculate_average_of_10_times_ten-foldCV.py
```

### Another usage:

> The code has been structured so that you can clearly see the result of each step. Therefore, the following usage is also feasible. But it is recommended to use the above usage

1. Set the parameters in 'src/config.py'

2. Divide the data

```cmd
python process_data.py
```

3. The features are partitioned according to the partitioned data

```cmd
python process_feature.py
```

4. Encoding features and data
```cmd
python get_distribution.py
```
5. Training GAN
```cmd
python train.py
```
6. Compute the average of 10 results of ten-fold cross-validation. Note: It is necessary to repeat steps 1-5 10 times to calculate the average of 10 iterations of ten-foldCVs
```cmd
python calculate_average_of_10_times_ten-foldCV.py
```

### File descriptions:

> To make the code look clearer and easier to read, we structured the code and related files. The output of each module of the framework is stored in a persistent form on disk, which is helpful for module debugging. You can clearly see that these intermediate data are in the directory (data/partitioned_data and results/). In addition, ablation experiments can be performed more conveniently.

A rough description of the file structure is listed below, more details can be seen in /directory_structure.txt:

```plain
----GGANDTI\        
    |----cache\                             Cache, the constructed heterogeneous network is serialized into binary objects and stored here
    |----data\        
    |    |----partitioned_data\             The divided data is stored here. These include the training set, test set, and feature distribution used to initialize.
    |    |----RawData\                      Raw data
    |----results\                           The results of training and computational validation are saved here
    |----saved_model\                       The saved model can be reloaded from here
    |----src\                               Code is stored under this folder
    |    |----p1_preprocessing_data\        Divide training set and test set
    |    |    |----load_data.py             Load the raw data
    |    |    |----process_data.py          The operations of partitioning data and constructing heterogeneous networks are written here
    |    |    |----utils.py                 
    |    |----p2_preprocessing_feature\     According to the divided training set and test set, to divide the features
    |    |    |----load_feature.py          Load the original feature matrix
    |    |    |----process_feature.py       Divide and save operations are written here
    |    |----p3_get_latent_variable\       The code corresponding to the encoder is written here
    |    |    |----get_distribution.py      Encoder construction and encoding process. The encoded result is saved in results/e/0fold/e_gen_.emb
    |    |    |----encoder.py               The model vgae used by the encoder
    |    |    |----utils.py                 
    |    |----p4_GAN\                       The core part of the framework, including GAN training and validation
    |    |    |----link_prediction.py       Evaluate GANs
    |    |    |----model.py                 The GAN model is defined here
    |    |    |----train.py                 GAN training and validation starts here
    |    |    |----utils.py
    |    |----p5_other\                     Please pay attention to distinguish between "10-fold CVs" and "10 times 10-flod CVs". "10-fold CVs" is to divide the data into 10 parts, and take turns to do training set and validation set. "10 times 10-flod CVs" refers to is to repeat the previous operation 10 times
    |    |    |----calculate_average_of_10_times_10-foldCV.py   Calculate the mean of 10 iterations of the 10-fold CV. See usage step 4.
    |    |----config.py                     By setting different seeds ten times, 10 times 10-fold divisions can be obtained. Each 10-fold CVs can get 10 groups of divided data, which can be used to complete a 10-fold CV.
    |    |----run.py                        A script that completes the framework training and validation process with one click.
    |----directory_structure.txt            Detailed directory structure
    |----ReadMe.md                          README
    |----requirements.txt                   Python environment requirements.
```

### Requirements:

    pip install -r requirements.txt

### Datasets：

If you want to use other data, you have to provide or construct

- an N by N adjacency matrix (N is the number of nodes), and
- an N by D feature matrix (D is the number of features per node) -- optional

Have a look at the src/p1_preprocessing_data/load_data.py and src/p2_preprocessing_feature/load_feature.py for an example.

In this example, gold standard datasets of Yamanishi et al. is available on (http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/) and datasets of Luo et al. is available on
( https://github.com/luoyunan/DTINet).  
Note: drugs and proteins are organized in the same order across all files, including name lists, ID mappings and interaction/similarity matrices.

### Acknowledgements:

Thanks to Yamanish et al. and Luo at al. for the datasets,
which helped the research to proceed smoothly.
