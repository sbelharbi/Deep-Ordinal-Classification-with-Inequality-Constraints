#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""criteria.py
Implements different learning losses and metrics.
"""
import numpy as np
from scipy.stats import norm

import torch
import torch.nn as nn
from torch.nn import functional as F


__all__ = ["LossCE", "LossPN", "LossELB", "LossRLB", "LossREN", "LossLD",
           "LossMV", "LossPO", "Metrics"]


class _NeighborDifferentiator(nn.Module):
    """
    Computes the difference between the neighbors: `Delta_a^b(s)`.
    use 1D convolution for ease, clean, and speed.

    The differences are computed over all the adjacent neighbors, from left
    to right. i.e., s(i) - s(i+1).

    If s in R^c, the output of this differentiator is in R^(c-1).
    It operates over a 2D matrix over rows (teh convolution is performed over
    the rows).
    """
    def __init__(self):
        """
        Init. function.
        """
        super(_NeighborDifferentiator, self).__init__()
        # constant convolution kernel.
        kernel2rightct = torch.tensor(
            [+1, -1], requires_grad=False).float().view(1, 1, 2)
        self.register_buffer("kernel2right", kernel2rightct)

    def forward(self, inputs):
        """
        Compute the difference between all the adjacent neighbors.
        :param inputs: torch tensor of size (nbr_samples, nbr_calsses)
        contains scores.
        :return: left to right differences. torch tensor of size
        (nbr_samples, nbr_calsses - 1).
        """
        msg = "`inputs` must be a matrix with size (h, w) where `h` is the " \
              "number of samples, and `w` is the number of classes. We found," \
              " `inputs.ndim`={}, and `inputs.shape`={} .... " \
              "[NOT OK]".format(inputs.ndim, inputs.shape)
        assert inputs.ndim == 2, msg

        assert inputs.shape[1] > 1, "Number of classes must be > 1 ....[NOT OK]"
        h, w = inputs.shape
        output2right = F.conv1d(
            input=inputs.view(h, 1, w), weight=self.kernel2right, bias=None,
            stride=1, padding=0, dilation=1, groups=1)
        msg = "Something's wrong. We expected size of 1d convolution to be " \
              "{}. but, we found {}.... [NOT OK]".format(
               inputs.shape, output2right.shape)
        assert output2right.shape == (h, 1, w - 1), msg
        return output2right.squeeze()

    def __str__(self):
        return "{}(): module that computes the difference between all " \
               "the adjacent neighbors from left to right using 1D " \
               "convolution, s(i) - s(i+1). Operates on 2D matrix.".format(
                self.__class__.__name__)


class _RightAndLeftDelta(nn.Module):
    """
    Computes the Delta on the left and the right of a reference label.
    It is the big Delta referred to in the paper.
    Then, put everything in one single matrix (or a row for one single sample).
    """
    def __init__(self):
        """
        Init. function.
        """
        super(_RightAndLeftDelta, self).__init__()
        self.differentiator = _NeighborDifferentiator()

    def forward(self, inputs, rlabels):
        """
        Compute the difference between neighbors on the left and right. Then,
        put all the differences within the same vector.
        :param inputs: torch tensor of size (nbr_samples, nbr_calsses)
        contains scores.
        :param rlabels: torch.long tensor. Reference labels to which we
        need to compute left and right differences. It can be the ground
        truth, the predicted labels, or any other labels.
        :return: left and right differences in one single matrix. torch
        tensor of size (nbr_samples, nbr_calsses - 1).
        """
        msg = "`inputs` must be a matrix with size (h, w) where `h` is the " \
              "number of samples, and `w` is the number of classes. We found," \
              " `inputs.ndim`={}, and `inputs.shape`={} .... " \
              "[NOT OK]".format(inputs.ndim, inputs.shape)
        assert inputs.ndim == 2, msg

        assert inputs.shape[1] > 1, "Number of classes must be > 1 ....[NOT OK]"

        assert rlabels.ndim == 1, "`rlabels` must be a vector ....[NOT OK]"
        msg = "`rlabels` and `inputs` dimension mismatch....[NOT OK]"
        assert rlabels.numel() == inputs.shape[0], msg
        y = rlabels.view(-1, 1)

        seeleft = self.differentiator(inputs).squeeze().view(inputs.shape[0],
                                                             -1)
        seeright = - seeleft

        h, w = seeleft.shape
        yy = y.repeat(1, w).float()
        idx = torch.arange(start=0, end=w, step=1, dtype=seeleft.dtype,
                           device=seeleft.device, requires_grad=False)
        idx = idx.repeat(h, 1)
        # ======================================================================
        #                       LEFT TO RIGHT
        # ======================================================================
        leftonrightoff = (idx < yy).type(seeleft.dtype).to(
            seeleft.device).requires_grad_(False)
        leftside = leftonrightoff * seeleft

        # ======================================================================
        #                       RIGHT TO LEFT
        # ======================================================================
        leftoffrighton = 1 - leftonrightoff  # idx > yy or idx == yy.
        rightside = leftoffrighton * seeright

        # ======================================================================
        #                  BOTH SIDES IN ONE SINGLE MATRIX
        # ======================================================================
        return leftside + rightside

    def __str__(self):
        return "{}(): module that computes the difference between all " \
               "the adjacent neighbors from left to right and right to left " \
               "using 1D " \
               "convolution. Operates on 2D matrix." \
               "Then, puts all the differences in the same row" \
               "(for each input row)".format(self.__class__.__name__)


class _CE(nn.Module):
    """
    Cross-entropy loss.
    """
    def __init__(self):
        """
        Init. function.
        """
        super(_CE, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, scores, labels):
        """
        Forward function.
        :param scores: matrix (n, nbr_c) of unormalized scores.
        :param labels: vector of Log integers. The ground truth labels.
        :return: real value cross-entropy loss.
        """
        assert scores.shape[1] > 1, "Number of classes must be > 1 ....[NOT OK]"

        return self.loss(scores, labels)

    def __str__(self):
        return "{}(): standard cross-entropy method".format(
            self.__class__.__name__)


class _Penalty(nn.Module):
    """
    Penalty based method (PN).
    """
    def __init__(self, lamb=1e-5, eps=1e-1):
        """
        Init. function.
        :param lamb: float > 0. Lambda.
        :param eps: float > 0. the slack. Since we operate over the scores,
        and not on probabilities, it is ok that `eps` is big.
        """
        super(_Penalty, self).__init__()

        assert isinstance(eps, float), "`eps` must be a float. You provided" \
                                       ": {} .... [NOT OK]".format(type(eps))
        assert eps > 0., "`eps` must be > 0. You provided: {} ...." \
                         "[NOT OK]".format(eps)

        assert isinstance(lamb, float), "`lamb` must be a float. You provided" \
                                        ": {} .... [NOT OK]".format(type(lamb))
        assert eps > 0., "`lamb` must be > 0. You provided: {} ...." \
                         "[NOT OK]".format(lamb)

        self.register_buffer(
            "eps", torch.tensor([eps], requires_grad=False).float())
        self.register_buffer(
            "lamb", torch.tensor([lamb], requires_grad=False).float())

        self.all_delta = _RightAndLeftDelta()

    def forward(self, scores, labels):
        """
        Forward function.
        :param scores: matrix (n, nbr_c) of unormalized scores.
        :param labels: vector of Log integers. The ground truth labels.
        :return: real value penalty-based loss.
        """
        assert scores.shape[1] > 1, "Number of classes must be > 1 ....[NOT OK]"

        diff_matrix = self.all_delta(scores, labels)
        # penalise values that are >= 0.
        positive = ((diff_matrix > 0) | (diff_matrix == 0)).float()

        mtx_loss = positive * torch.pow(positive * diff_matrix - self.eps, 2)
        loss = mtx_loss.sum(dim=1).mean()
        return self.lamb * loss

    def __str__(self):
        return "{}(): penalty-based method.".format(self.__class__.__name__)


class _ExtendedLB(nn.Module):
    """
    Extended log-barrier method (ELB).
    """
    def __init__(self, init_t=1., max_t=10., mulcoef=1.01):
        """
        Init. function.

        :param init_t: float > 0. The initial value of `t`.
        :param max_t: float > 0. The maximum allowed value of `t`.
        :param mulcoef: float > 0.. The coefficient used to update `t` in the
        form: t = t * mulcoef.
        """
        super(_ExtendedLB, self).__init__()

        msg = "`mulcoef` must be a float. You provided {} ....[NOT OK]".format(
            type(mulcoef))
        assert isinstance(mulcoef, float), msg
        msg = "`mulcoef` must be > 0. float. You provided {} " \
              "....[NOT OK]".format(mulcoef)
        assert mulcoef > 0., msg

        msg = "`init_t` must be a float. You provided {} ....[NOT OK]".format(
            type(init_t))
        assert isinstance(init_t, float), msg
        msg = "`init_t` must be > 0. float. You provided {} " \
              "....[NOT OK]".format(init_t)
        assert init_t > 0., msg

        msg = "`max_t` must be a float. You provided {} ....[NOT OK]".format(
            type(max_t))
        assert isinstance(max_t, float), msg
        msg = "`max_t` must be > `init_t`. float. You provided {} " \
              "....[NOT OK]".format(max_t)
        assert max_t > init_t, msg

        self.register_buffer(
            "mulcoef", torch.tensor([mulcoef], requires_grad=False).float())
        # create `t`.
        self.register_buffer(
            "t_lb", torch.tensor([init_t], requires_grad=False).float())

        self.register_buffer(
            "max_t", torch.tensor([max_t], requires_grad=False).float())

        self.all_delta = _RightAndLeftDelta()

    def set_t(self, val):
        """
        Set the value of `t`, the hyper-parameter of the log-barrier method.
        :param val: float > 0. new value of `t`.
        :return:
        """
        msg = "`t` must be a float. You provided {} ....[NOT OK]".format(
            type(val))
        assert isinstance(val, float) or (isinstance(val, torch.Tensor) and
                                          val.ndim == 1 and
                                          val.dtype == torch.float), msg
        msg = "`t` must be > 0. float. You provided {} ....[NOT OK]".format(val)
        assert val > 0., msg

        if isinstance(val, float):
            self.register_buffer(
                "t_lb", torch.tensor([val], requires_grad=False).float()).to(
                self.t_lb.device
            )
        elif isinstance(val, torch.Tensor):
            self.register_buffer("t_lb", val.float().requires_grad_(False))

    def update_t(self):
        """
        Update the value of `t`.
        :return:
        """
        self.set_t(torch.min(self.t_lb * self.mulcoef, self.max_t))

    def forward(self, scores, labels):
        """
        The forward function.
        :param scores: matrix (n, nbr_c) of unormalized scores.
        :param labels: vector of Log integers. The ground truth labels.
        :return: real value extended-log-barrier-based loss.
        """
        assert scores.shape[1] > 1, "Number of classes must be > 1 ....[NOT OK]"

        diff_matrix = self.all_delta(scores, labels)
        loss_mtx = diff_matrix * 0.

        # vals <= -1/(t**2).
        ct = - (1. / (self.t_lb**2))

        idx_less = ((diff_matrix < ct) | (diff_matrix == ct)).nonzero()
        if idx_less.numel() > 0:
            val_less = diff_matrix[idx_less[:, 0], idx_less[:, 1]]
            loss_less = - (1. / self.t_lb) * torch.log(- val_less)
            loss_mtx[idx_less[:, 0], idx_less[:, 1]] = loss_less

        # vals > -1/(t**2).
        idx_great = (diff_matrix > ct).nonzero()
        if idx_great.numel() > 0:
            val_great = diff_matrix[idx_great[:, 0], idx_great[:, 1]]
            loss_great = self.t_lb * val_great - (1. / self.t_lb) * \
                torch.log((1. / (self.t_lb**2))) + (1. / self.t_lb)
            loss_mtx[idx_great[:, 0], idx_great[:, 1]] = loss_great

        loss = loss_mtx.sum(dim=1).mean()
        return loss

    def __str__(self):
        return "{}(): extended-log-barrier-based method.".format(
            self.__class__.__name__)


class _RectifiedLB(_ExtendedLB):
    """
    Rectified log-barrier method (RLB).
    """
    def __init__(self, init_t=1., max_t=10., mulcoef=1.1, epsp=1e-1):
        """
        Init. function.

        :param init_t: float > 0. The initial value of `t`.
        :param max_t: float > 0. The maximum allowed value of `t`.
        :param mulcoef: float > 0.. The coefficient used to update `t` in the
        form: t = t * mulcoef.
        :param epsp: float > 0. Float constant used to avoid rescaling the
        minimum negative value to -1.
        """
        super(_RectifiedLB, self).__init__(
            init_t=init_t, max_t=max_t, mulcoef=mulcoef)

        msg = "`epsp` must be a float. You provided {} ....[NOT OK]".format(
            type(epsp))
        assert isinstance(epsp, float), msg
        msg = "`epsp` must be > 0. float. You provided {} " \
              "....[NOT OK]".format(epsp)
        assert epsp > 0., msg

        self.register_buffer(
            "epsp", torch.tensor([epsp], requires_grad=False).float())

        self.register_buffer(
            "one", torch.tensor([1.], requires_grad=False).float())

        self.rectifier = nn.ReLU(inplace=False)

    def scale_negatives(self, inputs):
        """
        Scale negative values into ]-1, 0[.
        :param inputs: torch.tensor. Matrix with left and right differences
        all computed.
        :return: the input matrix with its negative values rescaled.
        """
        min_vals = inputs.min(dim=1)[0]  # a vector with shape (n)
        # do nothing for positive values (div by 1.)
        idx_pos = ((min_vals > 0.) | (min_vals == 0.)).nonzero()
        # there is nothing negative: then, do nothing.
        if idx_pos.numel() == inputs.shape[0]:
            return inputs
        # if there is something positive.
        if idx_pos.numel() > 0:
            min_vals[idx_pos[:, 0]] = self.one
        # scale negative values.
        idx_neg = (min_vals < 0.).nonzero()
        # there must be some negative values since we are here.
        assert idx_neg.numel() > 0, "Something's wrong. We expected some" \
                                    "negative values at this point." \
                                    "but, there seems none." \
                                    "Now, why is that?"

        dummy = torch.zeros_like(min_vals)
        dummy[idx_neg[:, 0]] = self.epsp
        min_vals = torch.abs(min_vals - dummy)
        # min_vals[idx_neg[:, 0]] = torch.abs(
        #     dummy - self.epsp)

        min_vals = min_vals.view(-1, 1).repeat(1, inputs.shape[1])
        position_neg_inputs = (inputs < 0.).float()
        min_vals = min_vals * position_neg_inputs + (self.one -
                                                     position_neg_inputs)
        # min_vals contains now: the value 1. for the positive or nul
        # differences, and some value for the negative differences.

        rescaled = inputs / min_vals
        assert rescaled.min() > -1, "WoW! min value of the rescaled" \
                                    "matrix are below -1. This means" \
                                    "one of these two things:" \
                                    "1. your self.epsp == 0." \
                                    "2. or, something is wrong." \
                                    "After the rescaling, the values should" \
                                    "be > -1. " \
                                    "found rescaled.min() = {}" \
                                    ".... [NOT OK]".format(
            rescaled.min())
        return rescaled

    def forward(self, scores, labels):
        """
        The forward function.
        :param scores: matrix (n, nbr_c) of unormalized scores.
        :param labels: vector of Log integers. The ground truth labels.
        :return: real value rectified-log-barrier-based loss.
        """
        assert scores.shape[1] > 1, "Number of classes must be > 1 ....[NOT OK]"

        diff_matrix = self.all_delta(scores, labels)
        # Scale negative values.
        # diff_matrix = self.scale_negatives(diff_matrix)
        # rectify the differences.
        # diff_matrix = self.rectifier(diff_matrix)

        # The rest is similar to extended-lb.
        loss_mtx = diff_matrix * 0.
        # vals <= -1/(t**2).
        ct = - (1. / (self.t_lb ** 2))

        idx_less = ((diff_matrix < ct) | (diff_matrix == ct)).nonzero()
        if idx_less.numel() > 0:
            val_less = diff_matrix[idx_less[:, 0], idx_less[:, 1]]
            loss_less = - (1. / self.t_lb) * torch.log(- val_less)
            loss_mtx[idx_less[:, 0], idx_less[:, 1]] = loss_less

        # vals > -1/(t**2).
        idx_great = (diff_matrix > ct).nonzero()
        if idx_great.numel() > 0:
            val_great = diff_matrix[idx_great[:, 0], idx_great[:, 1]]
            loss_great = self.t_lb * val_great - (1. / self.t_lb) * \
                         torch.log((1. / (self.t_lb ** 2))) + (1. / self.t_lb)
            loss_mtx[idx_great[:, 0], idx_great[:, 1]] = loss_great

        loss_mtx = self.rectifier(loss_mtx)
        loss = loss_mtx.sum(dim=1).mean()
        assert loss >= 0., "Something is wrong. loss={} .... [NOT OK]".format(
            loss
        )

        return loss

    def __str__(self):
        return "{}(): rectified-log-barrier-based method.".format(
            self.__class__.__name__)


# ==============================================================================
#                          PUBLIC LOSSES
#                 1. LossCE: Cross-entropy loss.
#                 2. LossPN: Penalty-based loss.
#                 3. LossELB: Extended log-barrier loss.
#                 4. LossRLB: Rectified log-barrier loss.
#                 5. LossREN: Re-encode loss.
#                 6. LossLD: Label distribution loss.
#                 7. LossMV: Mean-variance loss.
#                 8. LossPO: Hard-wired Poisson loss.
# ==============================================================================


class _DoNothingLoss(nn.Module):
    """
    Generic nn.Module module that does nothing.
    """
    def __init__(self):
        """
        Init. function.
        """
        super(_DoNothingLoss, self).__init__()

        self.register_buffer(
            "zero", torch.tensor([0.], requires_grad=False).float())

    def forward(self, scores, labels):
        """
        The forward function.
        :param scores: matrix (n, nbr_c) of unormalized scores.
        :param labels: vector of Log integers. The ground truth labels.
        :return: real value of the loss.
        """
        assert scores.shape[1] > 1, "Number of classes must be > 1 ....[NOT OK]"
        return self.zero


class _Loss(nn.Module):
    """
    Mother-class for loss.
    """
    def __init__(self):
        """
        Init. function.
        """
        super(_Loss, self).__init__()

        self.lossCE = _DoNothingLoss()  # cross-entropy loss
        self.lossCT = _DoNothingLoss()  # constraints loss

        self.t_tracker = []  # track `t` if there is any.

    def update_t(self):
        """
        Update t for log-barrier methods.
        :return:
        """
        if hasattr(self.lossCT, 'update_t'):
            self.t_tracker.append(self.lossCT.t_lb.item())
            self.lossCT.update_t()

    def forward(self, scores, labels):
        """
        The forward function.
        :param scores: matrix (n, nbr_c) of unormalized scores.
        :param labels: vector of Log integers. The ground truth labels.
        :return: real value of the loss.
        """
        assert scores.shape[1] > 1, "Number of classes must be > 1 ....[NOT OK]"

        return self.lossCE(scores=scores, labels=labels) + self.lossCT(
            scores=scores, labels=labels)

    def predict_label(self, scores):
        """
        Predict the output label based on the scores.
        This may differ from method to another. The default is argmax. But
        some methods do it differently. They have to override this function.
        :param scores: matrix (n, nbr_c) of unormalized scores.
        :return: vector of long integer. The predicted label(s).
        """
        return scores.argmax(dim=1, keepdim=False)

    @property
    def literal(self):
        """
        Name of the loss.
        :return: str, name of the loss.
        """
        return "Mother-loss"


class LossCE(_Loss):
    """
    Public cross-entropy loss (CE).
    """
    def __init__(self):
        """
        Init. function
        """
        super(LossCE, self).__init__()

        self.lossCE = _CE()  # cross-entropy loss
        self.lossCT = _DoNothingLoss()  # constraints loss

    def __str__(self):
        return "{}(): standard cross-entropy method".format(
            self.__class__.__name__)

    @property
    def literal(self):
        """
        Name of the loss.
        :return: str, name of the loss.
        """
        return "Cross-entropy loss (CE)"


class LossPN(_Loss):
    """
    Public penalty based method (PN).
    """
    def __init__(self, lamb=1e-5, eps=1e-1):
        """
        Init. function
        :param lamb: float > 0. Lambda.
        :param eps: float > 0. the slack. Since we operate over the scores,
        and not on probabilities, it is ok that `eps` is big.
        """
        super(LossPN, self).__init__()

        self.lossCE = _CE()  # cross-entropy loss
        self.lossCT = _Penalty(lamb=lamb, eps=eps)  # constraints loss

    def __str__(self):
        return "{}(): penalty-based method".format(
            self.__class__.__name__)

    @property
    def literal(self):
        """
        Name of the loss.
        :return: str, name of the loss.
        """
        return "Penalty loss (PN)"


class LossELB(_Loss):
    """
    Public extended-log-barrier based method (ELB).
    """
    def __init__(self, init_t=1., max_t=10., mulcoef=1.01):
        """
        Init. function
        :param init_t: float > 0. The initial value of `t`.
        :param max_t: float > 0. The maximum allowed value of `t`.
        :param mulcoef: float > 0.. The coefficient used to update `t` in the
        form: t = t * mulcoef.
        """
        super(LossELB, self).__init__()

        self.lossCE = _CE()  # cross-entropy loss
        self.lossCT = _ExtendedLB(
            init_t=init_t, max_t=max_t, mulcoef=mulcoef)  # constraints loss

    def __str__(self):
        return "{}(): extended-log-barrier-based method".format(
            self.__class__.__name__)

    @property
    def literal(self):
        """
        Name of the loss.
        :return: str, name of the loss.
        """
        return "Extended log-barrier loss (ELB)"


class LossRLB(_Loss):
    """
    Public rectified-log-barrier based method (ELB).
    """
    def __init__(self, init_t=1., max_t=10., mulcoef=1.1, epsp=1e-1):
        """
        Init. function

        :param init_t: float > 0. The initial value of `t`.
        :param max_t: float > 0. The maximum allowed value of `t`.
        :param mulcoef: float > 0.. The coefficient used to update `t` in the
        form: t = t * mulcoef.
        :param epsp: float > 0. Float constant used to avoid rescaling the
        minimum negative value to -1.
        """
        super(LossRLB, self).__init__()

        self.lossCE = _CE()  # cross-entropy loss
        self.lossCT = _RectifiedLB(
            init_t=init_t, max_t=max_t, mulcoef=mulcoef, epsp=epsp)
        # constraints loss

    def __str__(self):
        return "{}(): rectified-log-barrier-based method".format(
            self.__class__.__name__)

    @property
    def literal(self):
        """
        Name of the loss.
        :return: str, name of the loss.
        """
        return "Rectified log-barrier loss (RLB)"


class LossREN(_Loss):
    """
    Public loss implementing:
    `J. Cheng, Z. Wang, and G. Pollastri. A neural network ap-
     proach to ordinal regression. In Int. Joint Con. on Neural
     Networks (IEEE World Congress on Computational Intelli-
     gence), 2008.`
    """
    def __init__(self, thrs=0.5):
        """
        Init function.
        :param thrs: float in ]0., 1.[. The threshold. use to threshold the
        probabilities.
        """
        super(LossREN, self).__init__()
        msg = "`thrs` should be float. found {} ...[NOT OK]".format(
            type(thrs))
        assert isinstance(thrs, float), msg
        msg = "`thrs` must be in ]0., 1.[. found {} ... [NOT OK]".format(thrs)
        assert 0. < thrs < 1., msg

        self.thrs = thrs
        self.sigmoid = nn.Sigmoid()

    def re_encode_label(self, scores, labels, c):
        """
        Re-encode an integer label into a binary vector.

        :param scores: matrix (n, nbr_c) of unormalized scores.
        :param labels: vector of Log integers. The ground truth labels.
        :param c: int, total number of labels.
        :return: vector of float. If the label of a sample is k, the index k
        of the output vector and all the indices below it have the value 1.
        The rest is set to zero.
        """
        nbr_s = labels.shape[0]
        sliceint = torch.arange(0, c, dtype=scores.dtype,
                                device=scores.device, requires_grad=False)

        out = sliceint.view(1, -1).repeat(nbr_s, 1)
        dupl_labels = labels.view(-1, 1).repeat(1, c).float()
        dupl_labels = dupl_labels.to(scores.device)
        out = ((out < dupl_labels) | (out == dupl_labels)).float()  # < or =.
        return out

    def forward(self, scores, labels):
        """
        Override forward.
        Implement the mean squarred error as a loss as used in the paper.

        :param scores: matrix (n, nbr_c) of unormalized scores.
        :param labels: vector of Log integers. The ground truth labels.
        :return: real value of the loss.
        """
        # re-encode
        code = self.re_encode_label(scores, labels, scores.shape[1])
        # MSE between sigmoid and the new code.
        x = self.sigmoid(scores)
        return ((x - code)**2).sum(dim=1).mean()

    def predict_label(self, scores):
        """
        Predict the label based on the scores.
        :param scores: matrix (n, nbr_c) of unormalized scores.
        :return: vector of long integer. The predicted label(s).
        """
        # Apply sigmoid.
        x = self.sigmoid(scores)
        x = ((x > self.thrs) | (x == self.thrs)).float()
        # The prediction is the previous index of the first zero.
        # We are going to iterate over the samples... yes, it is happening,
        # like it or not.
        nbr_s = scores.shape[0]
        c = scores.shape[1]
        pred = torch.zeros(nbr_s, dtype=torch.long, device=scores.device)
        for i in range(nbr_s):
            ind = (x[i, :] == 0.).nonzero()  # find the ind. of zeros.
            if ind.numel() > 0:  # if there are zeros.
                outind = ind[0][0] - 1
                if outind < 0:  # if it is the first element that is 0, well,
                    # we output 0 (the first label)
                    pred[i] = 0
                else:
                    pred[i] = outind
            else:  # if all ones, out the las label.
                pred[i] = c - 1
        return pred

    def __str__(self):
        return "{}(): Re-encode method.".format(
            self.__class__.__name__)

    @property
    def literal(self):
        """
        Name of the loss.
        :return: str, name of the loss.
        """
        return "Re-encode loss (REN)"


class LossLD(_Loss):
    """
    Public loss implementing label distrbution loss.
    The main idea: convert the hard target into a distribution (Gaussian)
    where the mean is the target, and the variance is provided.
    The loss is teh divergence between the network output and the Gaussian
    distribution. We consider the Kullback–Leibler divergence. Prediction is
    achieved using naive Bayes rule.

    See:
    [1] `X. Geng. Label distribution learning. IEEE Transactions on
         Knowledge and Data Engineering, 28, 2016.`
    [2] `B.-B. Gao, C. Xing, C.-W. Xie, J. Wu, and X. Geng. Deep
         label distribution learning with label ambiguity. IEEE Trans-
         actions on Image Processing, 26, 2017.`
    [3] `X. Geng, C. Yin, and Z.-H. Zhou. Facial age estimation by
         learning from label distributions. PAMI, 35, 2013.`
    [4] `Z. Huo, X. Yang, C. Xing, Y. Zhou, P. Hou, J. Lv, and X. Geng.Deep
         age distribution learning for apparent age estimation. InCVPR
         workshops, pages 17–24, 2016.`
    """
    def __init__(self, var=1.):
        """
        Init function.
        :param var: float in ]0., inf[. The variance of the Gaussian.
        """
        super(LossLD, self).__init__()
        msg = "`var` should be float. found {} ...[NOT OK]".format(
            type(var))
        assert isinstance(var, float), msg
        msg = "`var` must be in ]0., inf[. found {} ... [NOT OK]".format(var)
        assert var > 0., msg

        self.std = np.sqrt(var)  # we need the standard deviation.
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def re_encode_label(self, scores, labels):
        """
        Re-encode an integer label into a Gaussian distribution-like.

        :param scores: matrix (n, nbr_c) of unormalized scores. it is used
        only to extract the tensor properties: dtype, device, shape.
        :param labels: vector of Log integers. The ground truth labels.
        :return: vector of float where each label has a probability computed
        using a Gaussian distribution. The returned vector does not form a
        probability distribution. We need to renormalize it later.
        """
        nbr_s = scores.shape[0]
        c = scores.shape[1]
        out = torch.zeros(scores.shape, dtype=scores.dtype,
                          device=scores.device, requires_grad=False)
        # we are going to loop over each sample to convert it.
        for i in range(nbr_s):
            out[i, :] = torch.from_numpy(
                norm.pdf(
                    x=np.arange(start=0, stop=c, step=1),
                    loc=float(labels[i]),
                    scale=self.std
                ))

        return out

    def forward(self, scores, labels):
        """
        Override forward.
        Implement the divergence loss between the label distribution and the
        obtained scores.

        :param scores: matrix (n, nbr_c) of unormalized scores.
        :param labels: vector of Log integers. The ground truth labels.
        :return: real value of the loss.
        """
        # re-encode into a unormalized distribution.
        code = self.re_encode_label(scores, labels)
        # normalize the distribution.
        code = F.softmax(code, dim=1)
        # compute the log_softmax to prepare the input of kl.
        logsfmx = F.log_softmax(scores, dim=1)
        # kl loss
        return self.kl(logsfmx, code)

    def __str__(self):
        return "{}(): Label distribution method.".format(
            self.__class__.__name__)

    @property
    def literal(self):
        """
        Name of the loss.
        :return: str, name of the loss.
        """
        return "Label distribution loss (LD)"


class LossMV(_Loss):
    """
    Public loss implementing mean-variance loss.
    The main idea: consider the output label as a random variable that is
    estimated based on the expected value of the index of an array.
    The loss combines softmax, with extra loss composed of two terms: the
    expected value of the output variable and its variance.

    See:
    [1] `H. Pan, H. Han, S. Shan, and X. Chen. Mean-variance loss
         for deep age estimation from a face. In CVPR, 2018.`
    [2] `C. Beckham and C. Pal. A simple squared-error reformulation
         for ordinal classification. CoRR, abs/1612.00775, 2016.`
    """
    def __init__(self, lam1=0.2, lam2=0.05):
        """
        Init function.
        :param lam1: float in ]0., inf[. The importance coefficient of the
        mean loss.
        :param lam2: float in ]0., inf[. The importance coefficient of the
        variance loss.
        """
        super(LossMV, self).__init__()
        msg = "`lam1` should be float. found {} ...[NOT OK]".format(
            type(lam1))
        assert isinstance(lam1, float), msg
        msg = "`lam1` should be float. found {} ...[NOT OK]".format(
            type(lam2))
        assert isinstance(lam2, float), msg
        msg = "`lam1` must be in ]0., inf[. found {} ... [NOT OK]".format(lam1)
        assert lam1 > 0., msg
        msg = "`lam2` must be in ]0., inf[. found {} ... [NOT OK]".format(lam2)
        assert lam2 > 0., msg

        self.register_buffer(
            "lam1", torch.tensor([lam1], requires_grad=False).float()
        )
        self.register_buffer(
            "lam2", torch.tensor([lam2], requires_grad=False).float()
        )

        self.lossCE = _CE()  # cross-entropy loss

    def forward(self, scores, labels):
        """
        Override forward.
        Loss = cross-entropy + lam1 * mean_loss + lam2 * variance_loss.

        :param scores: matrix (n, nbr_c) of unormalized scores.
        :param labels: vector of Log integers. The ground truth labels.
        :return: real value of the loss.
        """
        # cross-entropy: scalar.
        ce = self.lossCE(scores=scores, labels=labels)

        # probability.
        prob = F.softmax(scores, dim=1)

        # Mean loss
        nbr_s, c = scores.shape
        sliceit = torch.arange(0, c, dtype=scores.dtype, device=scores.device,
                               requires_grad=False)
        grid = sliceit.view(1, -1).repeat(nbr_s, 1)
        mean = (grid * prob).sum(dim=1)  # vector of means.
        mean_loss = (1./2.) * ((mean - labels.float())**2).mean()

        # variance loss
        grid2 = mean.view(-1, 1).repeat(1, c)
        variance_loss = (prob * ((grid - grid2)**2)).sum(dim=1).mean()

        # Total loss
        return ce + self.lam1 * mean_loss + self.lam2 * variance_loss

    def predict_label(self, scores):
        """
        Predict the label based on the scores.
        label = round (Expected_val(indices)) = round(i * p(i)).
        since the labels are encoded from 0, we use the floor function to
        be able to predict the label 0.

        :param scores: matrix (n, nbr_c) of unormalized scores.
        :return: vector of long integer. The predicted label(s).
        """
        prob = F.softmax(scores, dim=1)
        nbr_s, c = scores.shape
        sliceit = torch.arange(0, c, dtype=scores.dtype, device=scores.device,
                               requires_grad=False)
        grid = sliceit.view(1, -1).repeat(nbr_s, 1)
        mean = (prob * grid).sum(dim=1)  # vector of means.

        return mean.floor().long()

    def __str__(self):
        return "{}(): Mean-variance method.".format(
            self.__class__.__name__)

    @property
    def literal(self):
        """
        Name of the loss.
        :return: str, name of the loss.
        """
        return "Mean-variance loss (MV)"


class LossPO(_Loss):
    """
    Public loss implementing Poisson loss.
    The main idea: Hard-code the output network to model a Poisson distribution.
    We use the expected value of the indices to predict the output label.

    See:
    [1] `C. Beckham and C. Pal. Unimodal probability distributions
         for deep ordinal classification. CoRR, abs/1705.05278, 2017.`
    """
    def __init__(self):
        """
        Init function.
        """
        super(LossPO, self).__init__()

        self.lossCE = _CE()  # cross-entropy loss

    def get_le_poisson(self, scores):
        """
        Hard-wire the Poisson distribution.
        :param scores:
        :return:
        """
        raise NotImplementedError("We found it is suitable to implement this"
                                  "within the model. Do not call this "
                                  "function. Nothing to do here."
                                  "See deeplearning.models.PoissonHead().")

    def forward(self, scores, labels):
        """
        Override forward.

        :param scores: matrix (n, nbr_c) of unormalized scores obtained using
        the pooling deeplearning.models.PoissonHead().
        :param labels: vector of Log integers. The ground truth labels.
        :return: real value of the loss.
        """
        # Nothing to do here. Everything has been done within the model's
        # pooling to provide scores after heating them up using tau.
        return self.lossCE(scores=scores, labels=labels)

    def predict_label(self, scores):
        """
        Predict the label based on the scores.
        label = round (Expected_val(indices)) = round(i * p(i)).
        since the labels are encoded from 0, we use the floor function to
        be able to predict the label 0.

        :param scores: matrix (n, nbr_c) of unormalized scores.
        :return: vector of long integer. The predicted label(s).
        """
        prob = F.softmax(scores, dim=1)
        nbr_s, c = scores.shape
        sliceit = torch.arange(0, c, dtype=scores.dtype, device=scores.device,
                               requires_grad=False)
        grid = sliceit.view(1, -1).repeat(nbr_s, 1)
        mean = (prob * grid).sum(dim=1)  # vector of means.

        return mean.floor().long()

    def __str__(self):
        return "{}(): Hard-wired Poisson method.".format(
            self.__class__.__name__)

    @property
    def literal(self):
        """
        Name of the loss.
        :return: str, name of the loss.
        """
        return "Hard-wired Poisson loss (PO)"

# ==============================================================================
#                                 METRICS
#                1. ACC: Classification accuracy. in [0, 1]. 1 is the best.
#                2. MAE: Mean absolute error. in R^+. 0 is the best.
#                3. SOI: Sides order index. in [0, 1]. 1 is the best.
# ==============================================================================

class Metrics(nn.Module):
    """
    Compute three different metrics over the model's prediction (scores) with
    respect to some reference label:
    1. ACC: Classification accuracy. in [0, 1]. 1 is the best.
    2. MAE: Mean absolute error. in R^+. 0 is the best.
    3. SOI: Sides order index. in [0, 1]. 1 is the best.
    SOI metric is computed with respect to the true label and also with
    respect to the predicted label.

    """
    def __init__(self):
        """
        Init. function.
        """
        super(Metrics, self).__init__()

        self.all_delta = _RightAndLeftDelta()

    def forward(self, scores, labels, tr_loss, avg=True):
        """
        The forward function.

        :param scores: matrix (n, nbr_c) of unormalized scores.
        :param labels: vector of Log integers. The ground truth labels.
        :param tr_loss: instance of a class inhereted from `_Loss`. We use it
        to compute the predicted label.
        :param avg: bool If True, the metrics are averaged
        by dividing by the total number of samples. Default if True.
        :return: torch.Tensor vector of size 4 containing the 4 metrics:
        output[0] = acc
        output[1] = mae
        output[2] = soi_y  # reference: true label.
        output[3] = soi_py # reference: predicted label.
        """
        msg = "`scores` must be a matrix with size (h, w) where `h` is the " \
              "number of samples, and `w` is the number of classes. We found," \
              " `scores.ndim`={}, and `inputs.shape`={} .... " \
              "[NOT OK]".format(scores.ndim, scores.shape)
        assert scores.ndim == 2, msg
        assert scores.shape[1] > 1, "Number of classes must be > 1 ....[NOT OK]"

        assert labels.ndim == 1, "`labels` must be a vector ....[NOT OK]"
        msg = "`labels` and `scores` dimension mismatch....[NOT OK]"
        assert labels.numel() == scores.shape[0], msg

        n, c = scores.shape

        # This class should not be included in any gradient computation.
        with torch.no_grad():
            plabels = tr_loss.predict_label(scores)  # predicted labels
            # 1. ACC
            acc = ((plabels - labels) == 0.).float().sum()

            # 2. MAE
            mae = (plabels - labels).abs().sum()

            # 3. SOI_y: SOI with respect to the true labels.
            mtx_soi_y = self.all_delta(inputs=scores, rlabels=labels)
            assert mtx_soi_y.shape[1] == c - 1, "Something's wrong...[NOT OK]"
            soi_y = (mtx_soi_y < 0.).float().sum(dim=1).div_(c - 1).sum()
            assert 0 <= (soi_y / float(n)) <= 1., "Something is wrong." \
                                                  ".... [NOT OK]"

            # 4. SOI_{hat_y}: SOI with respect to the predicted labels.
            mtx_soi_py = self.all_delta(inputs=scores, rlabels=plabels)
            assert mtx_soi_py.shape[1] == c - 1, "Something's wrong...[NOT OK]"
            soi_py = (mtx_soi_py < 0.).float().sum(dim=1).div_(c - 1).sum()
            assert 0 <= (soi_py / float(n)) <= 1., "Something is wrong." \
                                                   ".... [NOT OK]"

            # Output
            output = torch.zeros(size=(4,), dtype=torch.float,
                                 device=scores.device, requires_grad=False)
            output[0] = acc
            output[1] = mae
            output[2] = soi_y
            output[3] = soi_py

            if avg:
                output = output / float(n)
        return output

    def __str__(self):
        return "{}(): computes ACC, MAE, SOI metrics.".format(
            self.__class__.__name__)


# ==============================================================================
#                                 TEST
# ==============================================================================


def test_CE():
    loss = _CE()
    print("Testing {}".format(loss))
    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    loss.to(DEVICE)
    num_classes = 5
    b = 16
    scores = (torch.rand(b, num_classes)).to(DEVICE)
    labels = torch.randint(low=0, high=num_classes, size=(b,), dtype=torch.long
                           ).to(DEVICE)

    lossx = loss(scores, labels)
    print(lossx.item())


def test_NeighborDifferentiator():
    loss = _NeighborDifferentiator()
    print("Testing {}".format(loss))
    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    loss.to(DEVICE)
    num_classes = 5
    b = 3
    scores = (torch.rand(b, num_classes)).to(DEVICE)

    lossx = loss(scores)
    print(scores.shape)
    print(scores)
    print(lossx.shape)
    print(lossx)


def test_RightAndLeftDelta():
    loss = _RightAndLeftDelta()
    print("Testing {}".format(loss))
    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    loss.to(DEVICE)
    num_classes = 5
    b = 3
    scores = (torch.rand(b, num_classes)).to(DEVICE)
    labels = torch.randint(low=0, high=num_classes, size=(b,), dtype=torch.long
                           ).to(DEVICE)

    lossx = loss(scores, labels)
    print(scores.shape)
    print(scores)
    print(lossx.shape)
    print(lossx)


def test_Penalty_PN():
    loss = _Penalty(lamb=1e-5, eps=1e-1)
    print("Testing {}".format(loss))
    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    loss.to(DEVICE)
    num_classes = 5
    b = 1
    scores = (torch.rand(b, num_classes)).to(DEVICE)
    labels = torch.randint(low=0, high=num_classes, size=(b,), dtype=torch.long
                           ).to(DEVICE)

    lossx = loss(scores, labels)
    print(lossx.item())


def test_Extended_LB():
    loss = _ExtendedLB()
    print("Testing {}".format(loss))
    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    loss.to(DEVICE)
    num_classes = 5
    b = 1
    scores = (torch.rand(b, num_classes)).to(DEVICE)
    labels = torch.randint(low=0, high=num_classes, size=(b,), dtype=torch.long
                           ).to(DEVICE)

    lossx = loss(scores, labels)
    print(lossx.item())

    #  test update t
    for i in range(10):
        print("i= {}, \t t={}".format(i, loss.t_lb))
        loss.update_t()
        print("Loss value {}".format(loss(scores, labels).item()))


def test_Rectified_LB():
    loss = _RectifiedLB()
    print("Testing {}".format(loss))
    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    loss.to(DEVICE)
    num_classes = 5
    b = 1
    scores = (torch.rand(b, num_classes)).to(DEVICE)
    labels = torch.randint(low=0, high=num_classes, size=(b,), dtype=torch.long
                           ).to(DEVICE)

    lossx = loss(scores, labels)
    print(lossx.item())

    #  test update t
    for i in range(10):
        print("i= {}, \t t={}".format(i, loss.t_lb))
        loss.update_t()
        print("Loss value {}".format(loss(scores, labels).item()))


def test_LossREN():
    loss = LossREN(thrs=0.5)
    print("Testing {}".format(loss))
    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    loss.to(DEVICE)
    num_classes = 5
    b = 1
    scores = (torch.rand(b, num_classes)).to(DEVICE)
    labels = torch.randint(low=0, high=num_classes, size=(b,), dtype=torch.long
                           ).to(DEVICE)

    # test reencoding.
    code = loss.re_encode_label(scores, labels, num_classes)
    print(labels)
    print("*****")
    print(code)

    lossx = loss(scores, labels)
    print(lossx.item())
    print("Scores:{}".format(scores))

    #  test update t
    for i in range(10):
        loss.update_t()
        print("Loss value {}".format(loss(scores, labels).item()))

    # test prediction
    t = torch.from_numpy(np.array([1, 1, -1, -1, 1]).astype(np.float32))
    predtrue = 1
    pred = loss.predict_label(t.view(1, -1))
    print("Test 1, {}. Exp. {}. found {}".format(
        pred == predtrue, predtrue, pred))

    t = torch.from_numpy(np.array([-1, -1, 1, 1]).astype(np.float32))
    predtrue = 0
    pred = loss.predict_label(t.view(1, -1))
    print("Test 2, {}. Exp. {}. found {}".format(
        pred == predtrue, predtrue, pred))

    t = torch.from_numpy(np.array([1, 1, 1, -1]).astype(np.float32))
    predtrue = 2
    pred = loss.predict_label(t.view(1, -1))
    print("Test 3, {}. Exp. {}. found {}".format(
        pred == predtrue, predtrue, pred))

    t = torch.from_numpy(np.array([1, 1, 1, 1]).astype(np.float32))
    predtrue = 3
    pred = loss.predict_label(t.view(1, -1))
    print("Test 4, {}. Exp. {}. found {}".format(
        pred == predtrue, predtrue, pred))

    t = torch.from_numpy(np.array([1, 1, -1, 1]).astype(np.float32))
    predtrue = 1
    pred = loss.predict_label(t.view(1, -1))
    print("Test 5, {}. Exp. {}. found {}".format(
        pred == predtrue, predtrue, pred))

    t = torch.from_numpy(np.array([-1, -1, -1, -1]).astype(np.float32))
    predtrue = 0
    pred = loss.predict_label(t.view(1, -1))
    print("Test 6, {}. Exp. {}. found {}".format(
        pred == predtrue, predtrue, pred))

    torch.manual_seed(1)
    t = torch.from_numpy(np.random.normal(size=(3, 4)))
    print(t)
    print("*******")
    print(loss.predict_label(t))


def test_LossLD():
    loss = LossLD(var=1.)
    print("Testing {}".format(loss))
    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    loss.to(DEVICE)
    num_classes = 5
    b = 1
    scores = (torch.rand(b, num_classes)).to(DEVICE)
    labels = torch.randint(low=0, high=num_classes, size=(b,), dtype=torch.long
                           ).to(DEVICE)

    # test reencoding.
    code = loss.re_encode_label(scores, labels)
    print("Label:", labels)
    print("***** Code")
    print(code)
    print("Probability:")
    prob = F.softmax(code, dim=1)
    print(prob)
    print(code[:, labels])
    print(prob[:, labels])

    lossx = loss(scores, labels)
    print("loss", lossx.item())
    print("Scores:{}".format(scores))
    print("pred: {}".format(loss.predict_label(scores)))

    #  test update t
    for i in range(10):
        loss.update_t()
        print("Loss value {}".format(loss(scores, labels).item()))


def test_LossMV():
    loss = LossMV()
    print("Testing {}".format(loss))
    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    loss.to(DEVICE)
    num_classes = 5
    b = 1
    scores = (torch.rand(b, num_classes)).to(DEVICE)
    labels = torch.randint(low=0, high=num_classes, size=(b,), dtype=torch.long
                           ).to(DEVICE)

    lossx = loss(scores, labels)
    print("labels", labels)
    print(lossx.item())
    print("Scores:{}".format(scores))
    print("Pred.labels", loss.predict_label(scores))


def test_LossPO():
    loss = LossPO()
    print("Testing {}".format(loss))
    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    loss.to(DEVICE)
    num_classes = 5
    b = 1
    scores = (torch.rand(b, num_classes)).to(DEVICE)
    labels = torch.randint(low=0, high=num_classes, size=(b,), dtype=torch.long
                           ).to(DEVICE)

    lossx = loss(scores, labels)
    print("labels", labels)
    print(lossx.item())
    print("Scores:{}".format(scores))
    print("Pred.labels", loss.predict_label(scores))


def test_public_losses():
    losses = [LossCE(), LossPN(), LossELB(), LossRLB(), LossREN(), LossLD(),
              LossMV(), LossPO()]

    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))

    for loss in losses:
        print("================= START TESTING ===================")
        print("Testing {}".format(loss))
        loss.to(DEVICE)
        num_classes = 5
        b = 10
        scores = (torch.rand(b, num_classes)).to(DEVICE)
        labels = torch.randint(low=0, high=num_classes, size=(b,),
                               dtype=torch.long).to(DEVICE)

        lossx = loss(scores, labels)
        print(lossx.item())

        #  test update t
        for i in range(10):
            if hasattr(loss.lossCT, 't_lb'):
                print("i= {}, \t t={}".format(i, loss.lossCT.t_lb))
            else:
                print("i= {}, \t t=NONE".format(i))
            loss.update_t()
            print("Loss value {}".format(loss(scores, labels).item()))

        print("================= END TESTING ===================")


def test_Metrics():
    metr = Metrics()
    print("Testing {}".format(metr))
    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    metr.to(DEVICE)
    num_classes = 4
    b = 1
    scores = (torch.rand(b, num_classes)).to(DEVICE)
    labels = torch.randint(low=0, high=num_classes, size=(b,), dtype=torch.long
                           ).to(DEVICE)
    labels *= 0.
    labels[0] = 0.
    scores *= 0.
    scores[0, 0] = 1.
    scores[0, 1] = 0.85
    scores[0, 2] = 0.75
    scores[0, 3] = 0.90

    for tr_loss in [LossCE(), LossPN(), LossELB(), LossRLB(), LossREN(),
                    LossLD(), LossMV(), LossPO()]:
        print("=============TESTING METRICS OVER {} ==========".format(
            tr_loss.literal))
        for avg in [True, False]:
            vals = metr(scores, labels, tr_loss, avg=avg)
            print("Avg: {}".format(avg))
            print(labels)
            print(scores)
            print(vals)


if __name__ == "__main__":
    torch.manual_seed(0)
    # test_CE()
    # test_NeighborDifferentiator()
    # test_RightAndLeftDelta()
    # test_Penalty_PN()

    # test_Extended_LB()
    # torch.manual_seed(0)
    # test_Rectified_LB()

    # test_LossREN()

    # test_LossLD()

    # test_LossMV()

    # test_LossPO()

    # test_public_losses()

    test_Metrics()

