from torch.optim.optimizer import Optimizer
import torch
from time import time

last_whole_model_params_list = []
new_whole_model_params_list = []
batch_cos_list = []
near_minimum = False

class MaliciousSGD(Optimizer):

    def __init__(self, params, lr=1e-2, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, gamma_lr_scale_up=1.0, min_grad_to_process=1e-4):

        self.lastrms_parameters_grads = []
        self.gamma_lr_scale_up = gamma_lr_scale_up
        self.min_grad_to_process = min_grad_to_process
        self.min_ratio = 1.0
        self.max_ratio = 5.0
        self.beta = 0.9


        self.certain_grad_ratios = torch.tensor([])

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(MaliciousSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MaliciousSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        currentrms_parameters_grads = 0

        loss = None
        if closure is not None:
            loss = closure()

        id_group = 0
        for i in range(len(self.param_groups)):
            self.lastrms_parameters_grads.append([])

        for group in self.param_groups:

            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            start = time()
            id_parameter = 0

            for p in group['params']:
                if p.grad is None:
                    continue

                if weight_decay != 0:
                    p.grad.data.add_(weight_decay, p.data)
                    # p.grad.data.add_(p.data, weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(p.grad.data).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, p.grad.data)
                        # buf.mul_(momentum).add_(p.grad.data, 1 - dampening)
                    if nesterov:
                        p.grad.data = p.grad.data.add(momentum, buf)
                    else:
                        p.grad.data = buf

                if not near_minimum:
                    if len(self.lastrms_parameters_grads[id_group]) <= id_parameter:
                        self.lastrms_parameters_grads[id_group].append((p.grad).clone().detach())
                    else:
                        lastrms_parameters_grads = self.lastrms_parameters_grads[id_group][id_parameter]
                        current_parameter_grad = p.grad.clone().detach()
                        currentrms_parameters_grads = self.beta * lastrms_parameters_grads + (1 - self.beta) * (current_parameter_grad)
                        ratio_grad_scale_up = 1.0 + self.gamma_lr_scale_up * (current_parameter_grad / (currentrms_parameters_grads + 1e-7))
                        # ratio_grad_scale_up = 1.0 + self.gamma_lr_scale_up * (currentrms_parameters_grads / (current_parameter_grad + 1e-7))
                        ratio_grad_scale_up = torch.clamp(ratio_grad_scale_up, self.min_ratio, self.max_ratio)
                        p.grad.mul_(ratio_grad_scale_up)
                end = time()
                self.lastrms_parameters_grads[id_group][id_parameter] = currentrms_parameters_grads

                p.data.add_(-group['lr'], p.grad.data)
                # p.data.add_(p.grad.data, -group['lr'])

                id_parameter += 1
            id_group += 1

        return loss