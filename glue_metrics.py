try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score
    import os
    import numpy as np

    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn


if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()

    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }

    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }

    def glue_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(labels)
        if task_name == "cola":
            return {"mcc": matthews_corrcoef(labels, preds)}
        elif task_name == "sst-2":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mrpc":
            return acc_and_f1(preds, labels)
        elif task_name == "sts-b":
            return pearson_and_spearman(preds, labels)
        elif task_name == "qqp":
            return acc_and_f1(preds, labels)
        elif task_name == "mnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mnli-m":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mnli-mm":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "qnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "rte":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "wnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "hans":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)

    def xnli_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(labels)
        if task_name == "xnli":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)

    GLUE_DATA_FOLDER = '/ssd2/GLUE/glue_data'

    def glue_metrics_file(task_name, pred_file):
        LABEL_POS = {
            'MRPC': 0,
            'CoLA': 1,
        }
        LABEL_MAPPING = {
            'MRPC': {'0': 0, '1': 1},
            'MNLI-m': {'contradiction': 0, 'entailment':1, 'neutral': 2},
            'MNLI-mm': {'contradiction': 0, 'entailment': 1, 'neutral': 2},
            'CoLA': {'0': 0, '1': 1},
            'SST-2': {'0': 0, '1': 1},
            'QQP': {'0': 0, '1': 1},
            'QNLI': {'entailment': 0, 'not_entailment': 1},
            'RTE': {'entailment': 0, 'not_entailment': 1},
        }
        SKIP = {
            'CoLA': 0
        }

        label_pos = -1
        if task_name in LABEL_POS:
            label_pos = LABEL_POS[task_name]

        if task_name in SKIP:
            skip = SKIP[task_name]
        else:
            skip = 1
        # task_name = 'RTE'
        # pred_file = '/ssd2/GLUE_TEST/RTE.tsv'
        if 'MNLI' in task_name:
            if '-mm' in task_name:
                dev_file = os.path.join(GLUE_DATA_FOLDER, 'MNLI', 'dev_mismatched.tsv')
            else:
                dev_file = os.path.join(GLUE_DATA_FOLDER, 'MNLI', 'dev_matched.tsv')
        else:
            dev_file = os.path.join(GLUE_DATA_FOLDER, task_name, 'dev.tsv')

        def load_file(file_name, label_pos, skip):

            with open(file_name) as f:
                info = f.readlines()
                info = np.array([i.split()[label_pos] for i in info[skip:]])
                if task_name in LABEL_MAPPING:
                    info = np.array([LABEL_MAPPING[task_name][i] for i in info])
                return info

        ground_truth = load_file(dev_file, label_pos, skip)
        predicted = load_file(pred_file, -1, 1)
        return glue_compute_metrics(task_name.lower(), predicted, ground_truth)
