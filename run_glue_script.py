import os
import argparse
import sys
import random
import json
import itertools
from collections import defaultdict
import glob
from multiprocessing import Pool


def run_process(proc):
    os.system(proc)


GLUE_DATA_FOLDER = '/ssd2/GLUE'
METRICS = {'CoLA': 'mcc', 'STS-B': 'pearson'}
hyper_parameter_name = ['--num-classes', '--total-num-update', '--warmup-updates', '--lr', '--max-sentences']
hyper_parameter_setting = {
    'MNLI':  [3, 123873, 7432],
    'QNLI':  [2, 33112,  1986],
    'QQP':   [2, 113272, 28318],
    'RTE':   [2, 2036,   122],
    'SST-2': [2, 20935,  1256],
    'MRPC':  [2, 2296,   137],
    'CoLA':  [2, 5336,   320],
    'STS-B': [1, 3598,   214]
}


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="", type=str)
parser.add_argument("--teacher_json", default="", type=str)
parser.add_argument("--student_json", default="", type=str)
parser.add_argument("--lr", default="", type=str)
parser.add_argument("--bs", default="", type=str)
parser.add_argument("--arch", default="roberta_medium", type=str)
parser.add_argument("--T", default="2", type=str)
parser.add_argument("--beta", default="0.5", type=str)
parser.add_argument("--crd_weight", default=None, type=str)
parser.add_argument('--nce_k', default=100, type=int)
parser.add_argument("--tasks", default="all", type=str)
parser.add_argument("--rounds", default=1, type=int)
parser.add_argument("--GPU", default='0,1,2,3', type=str)

args = parser.parse_args()

available_GPU = args.GPU.split(',')
ROBERTA_PATH = args.model_path
if args.tasks.lower() == 'all':
    tasks = ['RTE', 'MRPC', 'CoLA', 'SST-2', 'QNLI', 'MNLI', 'STS-B', 'QQP']
else:
    tasks = args.tasks.split(',')


ALL_TASKS = list(hyper_parameter_setting.keys())
for t in ALL_TASKS:
    if t not in tasks:
        hyper_parameter_setting.pop(t, None)

if len(args.teacher_json) > 0:
    if args.crd_weight is None:
        hyper_parameter_name += ['--temperature', '--kd_weight']
        MODE = 'KD'
        with open(args.teacher_json) as f:
            teacher_info = json.load(f)
        with open(args.student_json) as f:
            student_info = json.load(f)

        T = [float(l) for l in args.T.split(',')]
        beta = [float(l) for l in args.beta.split(',')]
        criterion = 'sentence_prediction_crd'
        for t in tasks:
            lr, bs = os.path.basename(student_info[t][0]).split('.')[0].split('_')
            hyper_parameter_setting[t] += [float(lr), int(bs), T, beta]
            arch = student_info[t][2]               # all archs needs to be the same

        OUTPUT_RES_FOLDER = '/ssd2/GLUE_results_pt_kd'
        OUTPUT_NAME_PARAS = ['--lr', '--max-sentences', '--temperature', '--kd_weight']
    else:
        hyper_parameter_name += ['--temperature', '--kd_weight', '--crd_weight']
        MODE = 'CRD'
        with open(args.teacher_json) as f:
            teacher_info = json.load(f)
        with open(args.student_json) as f:
            student_info = json.load(f)

        crd_weight = [float(l) for l in args.crd_weight.split(',')]
        criterion = 'sentence_prediction_crd'
        for t in tasks:
            tmp = os.path.basename(student_info[t][0]).split('_')
            lr, bs, T, beta = tmp[0], tmp[1], tmp[2], '.'.join(tmp[3].split('.')[:2])
            hyper_parameter_setting[t] += [float(lr), int(bs), T, beta, crd_weight]
            arch = student_info[t][2]  # all archs needs to be the same

        # OUTPUT_RES_FOLDER = '/ssd2/GLUE_CRD_results'
        OUTPUT_RES_FOLDER = f'/ssd2/GLUE_results_cls_crd_{args.nce_k}'
        OUTPUT_NAME_PARAS = ['--lr', '--max-sentences', '--temperature', '--kd_weight', '--crd_weight']
else:
    MODE = 'FT'
    teacher_info, student_info = None, None
    T, beta = None, None
    criterion = 'sentence_prediction'
    lr = [float(l) for l in args.lr.split(',')]
    bs = [int(l) for l in args.bs.split(',')]
    for t in tasks:
        hyper_parameter_setting[t] += [lr, bs]
    arch = args.arch
    OUTPUT_RES_FOLDER = '/ssd2/GLUE_results_extra2'
    OUTPUT_NAME_PARAS = ['--lr', '--max-sentences']

for k in hyper_parameter_setting:
    for idx in range(len(hyper_parameter_setting[k])):
        if not isinstance(hyper_parameter_setting[k][idx], list):
            hyper_parameter_setting[k][idx] = [hyper_parameter_setting[k][idx]]

if not os.path.isfile(ROBERTA_PATH):
    raise ValueError(f'{ROBERTA_PATH} not exist! please double check')


ROUND = args.rounds
N_GPU = len(available_GPU)
cur_gpu_idx = 0
all_cmds = defaultdict(list)
for r in range(ROUND):
    for t in tasks:
        base_cmd = f'python train.py {GLUE_DATA_FOLDER}/{t}-bin/ ' \
            f'--restore-file {ROBERTA_PATH} '\
            f'--max-positions 512 '\
            f'--max-tokens 4400 '\
            f'--task sentence_prediction '\
            f'--reset-optimizer --reset-meters '\
            f'--required-batch-size-multiple 1 '\
            f'--init-token 0 --separator-token 2 '\
            f'--arch  {arch} '\
            f'--criterion {criterion} '\
            f'--dropout 0.1 --attention-dropout 0.1 '\
            f'--weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 '\
            f'--clip-norm 0.0 '\
            f'--lr-scheduler polynomial_decay  '\
            f'--fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 '\
            f'--max-epoch 10 ' \
            f'--find-unused-parameters '\
            f'--s_dim_feat 4608 --t_dim_feat 4608 ' \
            f'--nce_k {args.nce_k} ' \
            f'--reset-dataloader '
        if MODE == 'CRD' or MODE == 'KD':
            teacher_checkpoint = teacher_info[t][0]
            base_cmd += f'--teacher_model_checkpoint {teacher_checkpoint} '\
                        f'--teacher_model_pt checkpoint_best.pt '\
                        f'--data_name_or_path {GLUE_DATA_FOLDER}/{t}-bin '
        if t != 'STS-B':
            base_cmd += '--maximize-best-checkpoint-metric '
            if t in METRICS:
                base_cmd += '--best-checkpoint-metric {} '.format(METRICS[t])
            else:
                base_cmd += '--best-checkpoint-metric accuracy '
        else:
            base_cmd += '--regression-target --best-checkpoint-metric loss'

        search_paras = list(itertools.product(*hyper_parameter_setting[t]))
        for sp in search_paras:
            additional_cmd = []
            name_para = []
            for i, (h, v) in enumerate(zip(hyper_parameter_name, sp)):
                additional_cmd.append(h + ' ' + str(v))
                #if 'lr' in h or 'max-sentences' in h:
                if h in OUTPUT_NAME_PARAS:
                    name_para.append(v)

            seed = random.randint(0, 2 ** 32 - 1)
            save_dir = os.path.join(OUTPUT_RES_FOLDER, t, '_'.join(os.path.dirname(ROBERTA_PATH).split('/')[-2:]),
                                    '_'.join([str(n) for n in name_para])+'.round.'+str(seed))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            log_file = os.path.join(save_dir, 'log.fairseq')
            cmd = base_cmd + f' --seed {seed} ' + ' '.join(additional_cmd) + f' --save-dir {save_dir} > {log_file} 2>&1'
            cmd += f'; python parse_log.py eval_log {log_file}'
            current_GPU = available_GPU[cur_gpu_idx]
            all_cmds[cur_gpu_idx].append(f"CUDA_VISIBLE_DEVICES={current_GPU} " + cmd)
            cur_gpu_idx += 1
            cur_gpu_idx %= N_GPU

run_cmd = [';'.join(all_cmds[k]) for k in all_cmds]

# for r in run_cmd:
#    print(r, '\n')
pool = Pool(processes=N_GPU)
pool.map(run_process, run_cmd)
