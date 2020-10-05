# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.models.roberta.model import RobertaModel


@register_criterion('masked_lm_distill')
class MaskedLmLossDistill(FairseqCriterion):
    def __init__(self, task):
        super().__init__(task)
        self.teacher_model = RobertaModel.from_pretrained(task.args.teacher_model_checkpoint).model
        if self.teacher_model is None:
            print('teacher model not initialized')

        # freeze teacher model anyway
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        param.requires_grad = True
        self.T = task.args.temperature
        self.beta = task.args.kd_weight
        self.kd_loss_func = torch.nn.KLDivLoss(reduction='sum')
        self.print_teacher_loss = False

    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
        masked_tokens = sample['target'].ne(self.padding_idx)

        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).
        if masked_tokens.device == torch.device('cpu'):
            if not masked_tokens.any():
                masked_tokens.fill_(True)
        else:
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )

        logits_student = model(**sample['net_input'], masked_tokens=masked_tokens)[0]
        with torch.no_grad():
            logits_teacher = self.teacher_model(**sample['net_input'], masked_tokens=masked_tokens)[0]
        targets = model.get_targets(sample, [logits_student])
        targets = targets[masked_tokens]

        loss_ce = modules.cross_entropy(
            logits_student.view(-1, logits_student.size(-1)),
            targets.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )
        if self.print_teacher_loss:
            loss_ce_teacher = modules.cross_entropy(
                logits_teacher.view(-1, logits_teacher.size(-1)),
                targets.view(-1),
                reduction='sum',
                ignore_index=self.padding_idx,
            )
        # KD loss below
        loss_kd = self.kd_loss_func(F.log_softmax(logits_student/self.T, dim=-1),
                                    F.softmax(logits_teacher / self.T, dim=-1)) * self.T**2

        loss = (1-self.beta) * loss_ce + self.beta * loss_kd

        sample_size = masked_tokens.int().sum()
        logging_output = {
            'loss': loss.data,
            'ce_loss': loss_ce.data,
            'kd_loss': loss_kd.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        if self.print_teacher_loss:
            logging_output['ce_loss_teacher'] = loss_ce_teacher
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        loss_ce = sum(log.get('ce_loss', 0) for log in logging_outputs)
        loss_kd = sum(log.get('kd_loss', 0) for log in logging_outputs)
        # print('debug info:   loss kd =', loss_kd)
        # print('debug info:   loss kd =', [log.get('ce_loss', 0) for log in logging_outputs])
        if 'ce_loss_teacher' in logging_outputs[0]:
            loss_ce_teacher = sum(log.get('ce_loss_teacher', 0) for log in logging_outputs)

        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('loss_ce', loss_ce / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('loss_kd', loss_kd / sample_size / math.log(2), sample_size, round=3)
        if 'ce_loss_teacher' in logging_outputs[0]:
            metrics.log_scalar('loss_ce_teacher', loss_ce_teacher / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
