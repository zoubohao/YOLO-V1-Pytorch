from torch.optim.lr_scheduler import _LRScheduler
import torch
import matplotlib.pyplot as plt

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

if __name__ == '__main__':
    epoch = 50
    LR =1e-4
    warmEpoch = 5
    multiplier = 10
    v = torch.zeros(10)
    optim = torch.optim.SGD([v], lr=LR)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epoch, eta_min=0, last_epoch=-1)
    scheduler = GradualWarmupScheduler(optim, multiplier=multiplier, total_epoch=warmEpoch, after_scheduler=cosine_scheduler)
    a = []
    b = []
    for epoch in range(1, epoch + 1):
        a.append(epoch)
        b.append(optim.param_groups[0]['lr'])
        print(epoch, optim.param_groups[0]['lr'])
        scheduler.step()

    plt.plot(a,b)
    plt.show()