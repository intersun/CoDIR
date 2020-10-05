import os
import sys
import argparse
from fairseq.models.roberta import RobertaModel


GLUE_DATA_DIR = '/ssd2/GLUE'
RAW_DATA_DIR = os.path.join(GLUE_DATA_DIR, 'glue_data')

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="", type=str)
parser.add_argument("--checkpoint_name", default="checkpoint_best.pt", type=str)
parser.add_argument("--task_name", default="", type=str)

cmd = '--model_path /ssd2/GLUE_results/RTE/roberta_medium_32-16-0.0007/2e-05_32.round.2541887044 --task_name RTE'
args = parser.parse_args(cmd.split())

data_dir = os.path.join(GLUE_DATA_DIR, args.task_name + '-bin')
data_dir_raw = os.path.join(RAW_DATA_DIR, args.task_name)

roberta = RobertaModel.from_pretrained(
    args.model_path,
    checkpoint_file=args.checkpoint_name,
    # data_name_or_path=f'{args.task_name}-bin'
    data_name_or_path=data_dir
)


label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
with open(os.path.join(data_dir_raw, 'dev.tsv')) as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[1], tokens[2], tokens[3]
        tokens = roberta.encode(sent1, sent2)
        prediction = roberta.predict('sentence_classification_head', tokens).argmax()
        prediction_label = label_fn(prediction)
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))