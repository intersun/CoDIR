# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import argparse

import torch
import numpy as np
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.models.roberta.model import RobertaModel
from crd.criterion import CRDLoss


@register_criterion('sentence_prediction_crd')
class SentencePredictionCriterionCRD(FairseqCriterion):

    def __init__(self, task, classification_head_name, regression_target):
        super().__init__(task)
        self.classification_head_name = classification_head_name
        self.regression_target = regression_target

        self.teacher_model = RobertaModel.from_pretrained(model_name_or_path=task.args.teacher_model_checkpoint,
                                                          checkpoint_file=task.args.teacher_model_pt,
                                                          data_name_or_path=task.args.data_name_or_path).model
        # freeze teacher model anyway
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        if self.teacher_model is None:
            print('teacher model not initialized')

        self.print_teacher_loss = False
        self.use_mse = False

        self.T = task.args.temperature
        self.beta = task.args.kd_weight
        self.crd_weight = task.args.crd_weight
        if task.args.use_mse:
            self.kd_loss_func = F.mse_loss
        else:
            self.kd_loss_func = torch.nn.KLDivLoss(reduction='sum')

        self.num_samples = -1
        if self.crd_weight > 0.0:
            self.add_train_cls_label(task)

            self.nce_k = self.task.args.nce_k
            opt = argparse.Namespace(
                s_dim=self.task.args.s_dim_feat,
                t_dim=self.task.args.t_dim_feat,
                feat_dim=self.task.args.crd_feat_dim,
                nce_k=self.nce_k,
                nce_m=0.5,
                nce_t=0.07,
                n_data=self.num_samples
            )
            self.crd_criterion = CRDLoss(opt)
            self.replace = self.nce_k >= min([len(n) for n in self.cls_negative])

    def add_train_cls_label(self, task):
        try:
            dataset = task.datasets['train']
            label = np.array([t['target'].item() for t in dataset])
            num_classes = len(np.unique(label))
            self.num_samples = len(label)
            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(self.num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for i in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

            self.cls_positive = np.asarray(self.cls_positive)
            self.cls_negative = np.asarray(self.cls_negative)
        except KeyError:
            raise ValueError('dataset does not have training')

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--classification-head-name',
                            default='sentence_classification_head',
                            help='name of the classification head to use')
        # fmt: on

    def sample_neg_idx(self, pos_idx, target):
        neg_idx = np.array([np.random.choice(self.cls_negative[t.item()], self.nce_k, replace=self.replace) for t in target])
        sample_idx = torch.cat([torch.unsqueeze(pos_idx, 1), torch.from_numpy(neg_idx).to(target.device)], dim=1)
        return sample_idx

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, 'classification_heads')
            and self.classification_head_name in model.classification_heads
        ), 'model must provide sentence classification head for --criterion=sentence_prediction'

        sample['net_input']['return_all_hiddens'] = True
        logits, extra = model(
            **sample['net_input'],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )
        with torch.no_grad():
            logits_teacher, extra_teacher = self.teacher_model(
                **sample['net_input'],
                features_only=True,
                classification_head_name=self.classification_head_name,
            )
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()

        if self.num_samples > 0:
            # hidden_stat = torch.cat([t.transpose(0, 1).mean(1) for t in extra['inner_states'][1:]], dim=1)
            # hidden_state_teacher = torch.cat([t.transpose(0, 1).mean(1) for t in extra_teacher['inner_states'][1::2]], dim=1)
            hidden_stat = torch.cat([t.transpose(0, 1)[:, 0] for t in extra['inner_states'][1:]], dim=1)
            hidden_state_teacher = torch.cat([t.transpose(0, 1)[:, 0] for t in extra_teacher['inner_states'][1::2]], dim=1)

            # attn = torch.cat([t.reshape(logits.shape[0], -1) for t in extra['attn']], dim=1)
            # attn_teacher = torch.cat([t.reshape(logits.shape[0], -1) for t in extra_teacher['attn'][::2]], dim=1)

            idx = sample['id']
            neg_idx = self.sample_neg_idx(idx, targets)
            loss_crd = self.crd_criterion(hidden_stat, hidden_state_teacher, idx, neg_idx)

        if not self.regression_target:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            loss_ce = F.nll_loss(lprobs, targets, reduction='sum')

            loss_kd = self.kd_loss_func(F.log_softmax(logits/self.T, dim=-1),
                                        F.softmax(logits_teacher/self.T, dim=-1)) * self.T ** 2

            _, preds = torch.max(lprobs, dim=1)
        else:
            logits = logits.view(-1).float()
            targets = targets.float()
            loss_ce = F.mse_loss(logits, targets, reduction='sum')
            loss_kd = F.mse_loss(logits, logits_teacher, reduction='sum')
            preds = logits

        loss = (1-self.beta) * loss_ce + self.beta * loss_kd
        if self.crd_weight > 0:
            loss = loss + self.crd_weight * loss_crd
        logging_output = {
            'loss': loss.data,
            'ce_loss': loss_ce.data,
            'kd_loss': loss_kd.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
            'preds': preds,
            'labels': targets
        }
        if self.crd_weight > 0:
            logging_output['crd_loss'] = self.crd_weight * loss_crd

        if not self.regression_target:
            preds = logits.argmax(dim=1)
            logging_output['ncorrect'] = (preds == targets).sum()

        if self.print_teacher_loss:
            with torch.no_grad():
                lprobs = F.log_softmax(logits_teacher, dim=-1, dtype=torch.float32)
                loss_ce_teacher = F.nll_loss(lprobs, targets, reduction='sum')
            logging_output['ce_loss_teacher'] = loss_ce_teacher
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        loss_ce = sum(log.get('ce_loss', 0) for log in logging_outputs)
        loss_kd = sum(log.get('kd_loss', 0) for log in logging_outputs)
        loss_crd = sum(log.get('crd_loss', 0) for log in logging_outputs)

        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('loss_ce', loss_ce / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('loss_kd', loss_kd / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('loss_crd', loss_crd / sample_size / math.log(2), sample_size, round=3)
        if 'ce_loss_teacher' in logging_outputs[0]:
            loss_ce_teacher = sum(log.get('ce_loss_teacher', 0) for log in logging_outputs)
            metrics.log_scalar('loss_ce_teacher', loss_ce_teacher / sample_size / math.log(2), sample_size, round=3)

        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy', 100.0 * ncorrect / nsentences, nsentences, round=1)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
