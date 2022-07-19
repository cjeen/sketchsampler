from torch.optim.lr_scheduler import LambdaLR


def get_decay_scheduler(
        optimizer, max_iters, exponent
):
    def lr_lambda(current_step):
        return (1 - current_step * 1.0 / max_iters) ** exponent

    return LambdaLR(optimizer, lr_lambda)
