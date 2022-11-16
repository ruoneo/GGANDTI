Predicting drug-target interactions based on improved graph generative adversarial network and variational graph auto-encoder
==============
A framework for drug-target interactions prediction.

### Usage:

1. Activate the Python environment. (We provide a configured conda environment that can be downloaded from here https://drive.google.com/drive/folders/1zPY78onQRRlNKwVwMzlQuNP4wy95Lf50?usp=sharing)
2. Set hyper-parameters in `src/config.py`
3. Make sure you have switched to the root directory of the project. Use the following commands to perform training and validation.

```cmd
python src/run.py
```

4. Steps 2 and 3 are a complete ten-fold CV. Set the random number seed in `src/config.py`, and repeat steps 2 and 3 10 times to get the results of 10 times of ten-fold CV. Each repetition of ten-fold CV generates a file (results/{dataset}/final_results/final_result_{0}.txt) that holds the results for each fold and the average of the ten folds. Finally, execute the following command to calculate the average.

```cmd
python src/p5_other/calculate_average_of_10_times_of_ten-foldCV.py
```
> Note: 
> - This project has been tested with ``win10+RTX3080+python3.7+cuda11.2+cudnn8.1.0+tf1.15``. If you are using a graphics card with Ampere architecture, such as RTX 3080, please download tensorflow 1.15.4 from https://github.com/NVIDIA/tensorflow. Nvidia maintains a tf1.x version to keep compatibility with ampere architecture graphics cards.
> - ten-fold CV: The dataset is divided into ten parts, and 9 of them are used as training data and 1 is used as test data in turn for experimentation.
> - 10 times of ten-fold CV: Repeat ten-fold CV 10 times with different random number seeds.

### Another usage:

> The code has been structured so that you can clearly see the result of each step. Therefore, the following usage is also feasible. But it is recommended to use the above usage.

1. Set the parameters in 'src/config.py'

2. Divide the data

```cmd
python src/p1_preprocessing_data/process_data.py
```

3. The features are partitioned according to the partitioned data

```cmd
python src/p2_preprocessing_feature/process_feature.py
```

4. Encoding features and data
```cmd
python src/p3_get_latent_variable/get_distribution.py
```
5. Training GAN
```cmd
python src/p4_GAN/train.py
```
6. Compute the average of 10 times of ten-fold cross-validation. Note: You need to repeat the previous operation 10 times with different random number seed to have 10 results to calculate the average.
```cmd
python src/p5_other/calculate_average_of_10_times_of_ten-foldCV.py
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
    |    |----p5_other\                     Please pay attention to distinguish between "10-fold CV" and "10 times of 10-flod CV". "10-fold CV" is to divide the data into 10 parts, and take turns to do training set and validation set. "10 times of 10-flod CV" refers to is to repeat the previous operation 10 times
    |    |    |----calculate_average_of_10_times_10-foldCV.py   Calculate the mean of 10 iterations of the 10-fold CV. See usage step 4.
    |    |----config.py                     By setting different seeds ten times, 10 times 10-fold divisions can be obtained. Each 10-fold CV can get 10 groups of divided data, which can be used to complete a 10-fold CV.
    |    |----run.py                        A script that completes the framework training and validation process with one click.
    |----directory_structure.txt            Detailed directory structure
    |----ReadMe.md                          README
    |----requirements.txt                   Python environment requirements.
```

### Requirements:
Python 3.7 is recommended.

    pip install -r requirements.txt

### Datasets：

If you want to use other data, you have to provide or construct

- an N by N adjacency matrix (N is the number of nodes), and
- an N by D feature matrix (D is the number of features per node) -- optional

Have a look at the `src/p1_preprocessing_data/load_data.py` and `src/p2_preprocessing_feature/load_feature.py` for an example.

In this example, gold standard datasets of Yamanishi et al. is available on (http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/) and datasets of Luo et al. is available on
( https://github.com/luoyunan/DTINet).  
Note: drugs and proteins are organized in the same order across all files, including name lists, ID mappings and interaction/similarity matrices.

### Acknowledgements:

Thanks to Yamanish et al. and Luo at al. for the datasets,
which helped the research to proceed smoothly.
