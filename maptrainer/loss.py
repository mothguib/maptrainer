from torch.nn.modules.loss import *


def load_loss(loss_name: str = "NLL") -> type:
    loss_class_name = loss_name + "Loss"

    return globals()[loss_class_name]


class _Loss(Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(_WeightedLoss, self).__init__(size_average)
        self.register_buffer('weight', weight)


class EmpNLLLoss(_WeightedLoss):
    r"""The "empowered" negative log likelihood cost. It is useful to train a
    classification problem with n classes where the incorrect ones are
    intentionally utterly outlying. Seeing NLLLoss for documentation    """

    def __init__(self, _log_factor=300, weight=None, size_average=True,
                 ignore_index=-100):
        super(EmpNLLLoss, self).__init__(weight, size_average)
        self.ignore_index = ignore_index
        self.log_factor = _log_factor

    @staticmethod
    def _assert_no_grad(variable):
        assert not variable.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these variables as volatile or not requiring gradients"

    def forward(self, _input, target):
        self._assert_no_grad(target)
        return self.log_factor * F.nll_loss(_input, target, self.weight,
                                            self.size_average,
                                            self.ignore_index)


