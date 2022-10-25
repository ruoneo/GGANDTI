from src import config

'''
This script is to calculate the average of 10 times of 10-fold CVs (note not the average of one 10-fold CVs).

What you need to know: 
Each 10-fold CVs contains 10 split sets of data. These 10 groups will be selected in turn as the test set and the remaining 9 groups will be used as the training set. 
This process of 10 repeats of training and validation is called a 10-fold cross validation once.
GAN models are trained and validated by performing a full 10-fold cross-validation without additional scripting or hyperparameter configuration. The results are stored in results/e/final_results.

If you only need to observe the result of a 10-fold CVs once, then you don't need to execute this script.

> Note: 
> - ten-fold CV: The dataset is divided into ten parts, and 9 of them are used as training data and 1 is used as test data in turn for experimentation.
> - 10 times of ten-fold CV: Repeat ten-fold CV 10 times with different random number seeds.

'''

for dataset in config.datasets:
    auroc_results = []
    auprc_results = []
    for fold in range(10):
        # Each 10-fold cross-validation result is recorded in the following folder so read from here.
        filename = "../../results/{}/final_results/final_result_{}.txt".format(dataset, fold)
        with open(filename, "r") as inf:
            lines = inf.readlines()[4:]
            for line, index in zip(lines, range(2)):
                line = line.split()[0]
                if index == 0:
                    value = float(line[14:])
                    auroc_results.append(value)
                if index == 1:
                    value = float(line[14:])
                    auprc_results.append(value)
    avg_roc_score = sum(auroc_results) / len(auroc_results)
    avg_prc_score = sum(auprc_results) / len(auprc_results)
    print("百分比:{:.0%},数据集{}的十次验证平均结果: avg_auroc={:.5f},avg_auprc={:.5f}".format(config.percent, dataset, avg_roc_score, avg_prc_score))