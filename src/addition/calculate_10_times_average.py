from src import config

for dataset in config.datasets:
    auroc_results = []
    auprc_results = []
    for fold in range(10):
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