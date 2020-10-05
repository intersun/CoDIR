import argparse
import json
import os
from fairseq.models.roberta import RobertaModel
import torch
from fairseq import tasks, utils
from glue_metrics import glue_metrics_file
import pdb


GLUE_DATA_DIR = '/ssd2/GLUE'


def get_tokens(line, roberta, task):
    tokens = line.strip().split('\t')
    if task == "RTE":
        sent1, sent2 = tokens[1], tokens[2]
        tokens = roberta.encode(sent1, sent2)
    elif task == "MRPC":
        sent1, sent2 = tokens[3], tokens[4]
        tokens = roberta.encode(sent1, sent2)
    elif task == "CoLA":
        sent = tokens[1]
        tokens = roberta.encode(sent)
    elif task == "SST-2":
        sent = tokens[1]
        tokens = roberta.encode(sent)
    elif task == "STS-B":
        sent1, sent2 = tokens[7], tokens[8]
        tokens = roberta.encode(sent1, sent2)
    elif task == "QQP":
        sent1, sent2 = tokens[1], tokens[2]
        tokens = roberta.encode(sent1, sent2)
    elif task == "QNLI":
        sent1, sent2 = tokens[1], tokens[2]
        tokens = roberta.encode(sent1, sent2)
    elif task == "WNLI":
        sent1, sent2 = tokens[1], tokens[2]
        tokens = roberta.encode(sent1, sent2)
    elif task == "MNLI-m":
        sent1, sent2 = tokens[-2], tokens[-1]
        tokens = roberta.encode(sent1, sent2)
    elif task == "MNLI-mm":
        sent1, sent2 = tokens[-2], tokens[-1]
        tokens = roberta.encode(sent1, sent2)
    elif task == "AX":
        sent1, sent2 = tokens[-2], tokens[-1]
        tokens = roberta.encode(sent1, sent2)
    else:
        print("Task {} undefined".format(task))
        exit()
    return tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chk-dir", default="checkpoints", type=str)
    parser.add_argument("--chk-fname", default="checkpoint_last.pt", type=str)
    parser.add_argument("--task", default="sentence_prediction", type=str)
    parser.add_argument("--dset", default="CoLA", type=str)
    parser.add_argument("--test-path", default="glue_data")
    parser.add_argument("--out-path", default="glue-test")
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--max-positions", default=512, type=int)
    parser.add_argument("--regression-target", default=False, type=bool)
    parser.add_argument("--dataset-impl", default=None)
    parser.add_argument("--init-token", default=0, type=int)
    parser.add_argument("--separator-token", default=2, type=int)
    parser.add_argument("--no-shuffle", default=True, type=int)
    parser.add_argument("--seed", default=1, help="Shall not be used. Placeholder")
    parser.add_argument("--truncate-sequence", default=False)
    parser.add_argument("--split", default='test')

    if False:
        cmd = '--out-path /ssd2/GLUE_TEST/ft ' \
              '--dset RTE ' \
              '--test-path glue_data ' \
              '--chk-fname checkpoint_best.pt ' \
              '--num-classes 2 ' \
              '--chk-dir /ssd2/GLUE_results/RTE/roberta_medium_32-16-0.0007/2e-05_32.round.2541887044 ' \
              '--split dev ' \
              '--batch-size 16'
        args = parser.parse_args(cmd.split())
    args = parser.parse_args()
    args.add_prev_output_tokens = False
    print(args)

    bin_task = "MNLI" if "MNLI" in args.dset or args.dset == "AX" else args.dset
    args.data = os.path.join(GLUE_DATA_DIR, bin_task+"-bin")
    print(args.data)
    roberta = RobertaModel.from_pretrained(
        args.chk_dir,
        checkpoint_file=args.chk_fname,
        data_name_or_path=args.data
    )

    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.target_dictionary.nspecial]
    )
    ncorrect, nsamples = 0, 0
    roberta.to('cuda')
    roberta.eval()
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    task = tasks.setup_task(args)

    if args.split == 'test':
        if args.dset == "MNLI-m":
            split = "test"
        elif args.dset == "MNLI-mm":
            split = "test1"
        else:
            split = "test"
    else:
        if args.dset == "MNLI-m":
            split = "valid"
        elif args.dset == "MNLI-mm":
            split = "valid1"
        else:
            split = "valid"

    task.load_dataset(split)
    itr = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=4400,
        max_sentences=args.batch_size,
        max_positions=512,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
    ).next_epoch_itr(shuffle=False)
    with torch.no_grad():
        if split == 'test' or split == 'test1':
            out_file_name = os.path.join(args.out_path, args.dset + ".tsv")
        else:
            if '1' in split:
                out_file_name = os.path.join(args.out_path, args.dset + "-" + split[:-1] + ".tsv")
            else:
                out_file_name = os.path.join(args.out_path, args.dset + "-" + split + ".tsv")
        out_fs = open(out_file_name, "w+")
        out_fs.write("id\tlabel\n")

        scores_list = []
        for batch in itr:

            tokens = batch['net_input']['src_tokens']
            ids = batch['id']
            if args.dset == "STS-B":
                prediction_label = roberta.predict('sentence_classification_head', tokens, return_logits=True)
                for index, pred in zip(ids, prediction_label):
                    out_fs.write("%d\t%f\n"%(index, pred.item()))
                scores_list.append(prediction_label)
            else:
                logit = roberta.predict('sentence_classification_head', tokens, return_logits=True)
                prediction = logit.argmax(dim=1)
                for n_, (index, pred) in enumerate(zip(ids, prediction)):
                    # prediction_label = label_fn(pred.item())
                    prediction_label = label_fn(pred)
                    out_fs.write("%d\t%s\n" % (index, prediction_label))
                scores_list.append(logit)
        scores_mat = torch.cat(scores_list, 0)
        # torch.save({"model_name": args.chk_dir, "scores": scores_mat}, os.path.join(args.out_path, "scores", "%s_scores_%d.pt"%(args.dset, args.num)))

    out_fs.close()
    if not 'test' in split:
        out_metric = open(os.path.join(args.out_path, args.dset + "-" + split + "-metric.json"), 'w')
        json.dump(glue_metrics_file(args.dset, out_file_name), out_metric, indent=2)
        out_metric.close()
