import math

import torch
import torch.optim as optimLib
import torch.nn as nn
import torch.nn.functional as F
from utils import AddBias

def _extract_patches(x, kernel_size, stride, padding):
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(
        x.size(0), x.size(1), x.size(2), x.size(3) * x.size(4) * x.size(5))
    return x


def compute_cov_a(a, classname, layer_info, fast_cnn):
    batch_size = a.size(0)

    if classname == 'Conv2d':
        if fast_cnn:
            a = _extract_patches(a, *layer_info)
            a = a.view(a.size(0), -1, a.size(-1))
            a = a.mean(1)
        else:
            a = _extract_patches(a, *layer_info)
            a = a.view(-1, a.size(-1)).div_(a.size(1)).div_(a.size(2))
    elif classname == 'AddBias':
        is_cuda = a.is_cuda
        a = torch.ones(a.size(0), 1)
        if is_cuda:
            a = a.cuda()

    return a.t() @ (a / batch_size)


def compute_cov_g(g, classname, layer_info, fast_cnn):
    batch_size = g.size(0)

    if classname == 'Conv2d':
        if fast_cnn:
            g = g.view(g.size(0), g.size(1), -1)
            g = g.sum(-1)
        else:
            g = g.transpose(1, 2).transpose(2, 3).contiguous()
            g = g.view(-1, g.size(-1)).mul_(g.size(1)).mul_(g.size(2))
    elif classname == 'AddBias':
        g = g.view(g.size(0), g.size(1), -1)
        g = g.sum(-1)

    g_ = g * batch_size
    return g_.t() @ (g_ / g.size(0))


def update_running_stat(aa, m_aa, momentum):
    # Do the trick to keep aa unchanged and not create any additional tensors
    m_aa *= momentum / (1 - momentum)
    m_aa += aa
    m_aa *= (1 - momentum)


class SplitBias(nn.Module):
    def __init__(self, module):
        super(SplitBias, self).__init__()
        self.module = module
        self.add_bias = AddBias(module.bias.data)
        self.module.bias = None

    def forward(self, input):
        x = self.module(input)
        x = self.add_bias(x)
        return x


class SO_KLD_WRM_Optimizer(optimLib.Optimizer):
    def __init__(self,
                 model,
                 lr_function = lambda epoch_n, iteration_n: 0.1 ,
                 momentum = 0.9,
                 stat_decay = 0.99,
                 kl_clip = 0.001,
                 damping = 1e-2,
                 weight_decay = 0,
                 fast_cnn = False,
                 Ts = 1,
                 Tf = 1):
        defaults = dict()

        def split_bias(module):
            for mname, child in module.named_children():
                if hasattr(child, 'bias'):
                    module._modules[mname] = SplitBias(child)
                else:
                    split_bias(child)

        split_bias(model)

        super(SO_KLD_WRM_Optimizer, self).__init__(model.parameters(), defaults)

        self.known_modules = {'Linear', 'Conv2d', 'AddBias'}

        self.modules = []
        self.grad_outputs = {}
        self.acc_stats = False

        self.model = model
        self._prepare_model()

        self.steps = 0

        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}

        self.momentum = momentum
        self.stat_decay = stat_decay

        # LR stuff
        self.epoch_number = 1
        self.lr_function = lr_function
        self.lr = self.lr_function(self.epoch_number, 0) # self.lambdda = 1.0/lr
        self.prev_lr = self.lr_function(self.epoch_number, 0)
        
        self.kl_clip = kl_clip
        self.damping = damping
        self.weight_decay = weight_decay

        self.fast_cnn = fast_cnn

        self.Ts = Ts
        self.Tf = Tf
        
        self.prev_grad_in_param_list_form = [0 * i.data for i in self.model.parameters()] #set g_{-1} = 0

        self.optim = optimLib.SGD(
            model.parameters(),
            lr = self.lr * (1 - self.momentum),
            momentum = self.momentum)

    def _save_input(self, module, input):
        if self.model.training and self.steps % self.Ts == 0:
            classname = module.__class__.__name__
            layer_info = None
            if classname == 'Conv2d':
                layer_info = (module.kernel_size, module.stride,
                              module.padding)

            aa = compute_cov_a(input[0].data, classname, layer_info,
                               self.fast_cnn)

            # Initialize buffers
            if self.steps == 0:
                self.m_aa[module] = aa.clone()

            update_running_stat(aa, self.m_aa[module], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        if self.acc_stats:
            classname = module.__class__.__name__
            layer_info = None
            if classname == 'Conv2d':
                layer_info = (module.kernel_size, module.stride,
                              module.padding)

            gg = compute_cov_g(grad_output[0].data, classname,
                               layer_info, self.fast_cnn)

            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = gg.clone()

            update_running_stat(gg, self.m_gg[module], self.stat_decay)

    def _prepare_model(self):
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                assert not ((classname in ['Linear', 'Conv2d']) and module.bias is not None), \
                                    "You must have a bias as a separate layer"

                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)

    def step(self, epoch_number, error_write_path):
        # update LR according to the schedule
        self.lr = self.lr_function(epoch_number, self.steps)
        for g in self.optim.param_groups:
            g['lr'] = self.lr_function(epoch_number, self.steps)
        lambda_frac = self.prev_lr/self.lr # = lambda_new/ lambda_old        
        
        # save pure new gradient g_k to use next step (k+1)
        grad_at_k_in_param_list_form  = [i.grad.data + 0.0 for i in self.model.parameters()] 
        
        # ammend current grad to be g_k - \rho g_{k-1}
        for p, current_old_grad in zip(self.model.parameters(), self.prev_grad_in_param_list_form):
                p.grad.data = p.grad.data - self.stat_decay * lambda_frac * current_old_grad
                
        # set g_{k-1} <- g_k
        self.prev_grad_in_param_list_form = grad_at_k_in_param_list_form
        
        # Add weight decay - putting it here means weight decay bit will also be preconditioned by the KFAC matrix
        if self.weight_decay > 0:
            for p in self.model.parameters():
                p.grad.data.add_(self.weight_decay, p.data)

        updates = {}
        for i, m in enumerate(self.modules):
            assert len(list(m.parameters())
                       ) == 1, "Can handle only one parameter at the moment"
            classname = m.__class__.__name__
            p = next(m.parameters())

            la = self.damping + self.weight_decay

            if self.steps % self.Tf == 0: 
                self.d_a[m], self.Q_a[m] = torch.symeig(
                    self.m_aa[m], eigenvectors=True)
                self.d_g[m], self.Q_g[m] = torch.symeig(
                    self.m_gg[m], eigenvectors=True) # computes the eigen decomposition of bar_A and G matrices

                self.d_a[m].mul_((self.d_a[m] > 1e-6).float())
                self.d_g[m].mul_((self.d_g[m] > 1e-6).float())

            if classname == 'Conv2d':
                p_grad_mat = p.grad.data.view(p.grad.data.size(0), -1)
            else:
                p_grad_mat = p.grad.data
            
            # Compute KFAC direction in current block based on eigendecomposition
            v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
            v2 = v1 / (
                self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + la) # basically adds self.damping to the eigenvalues product (for each matrix entry)
            v = self.Q_g[m] @ v2 @ self.Q_a[m].t()

            v = v.view(p.grad.data.size()) # store KFAC direction in right format - for current block
            updates[p] = v

        vg_sum = 0
        for p in self.model.parameters():
            v = updates[p]
            vg_sum += (v * p.grad.data * self.lr * self.lr).sum()

        if self.kl_clip >= 1e10: # if we set the clip threshold rly high, just don't clip!
            nu = 1
        else:
            nu = min(1, math.sqrt(self.kl_clip / vg_sum))
            if nu < 1:
                print('KFAC STANDARD CLIP ACTIVATED!')

        for p in self.model.parameters():
            v = updates[p]
            p.grad.data.copy_(v)
            p.grad.data.mul_(nu) # We are multiplying by NU here, which clips the KFAC direction !!!
        
        # Save lr in prev_lr
        self.prev_lr = self.lr + 0.0
        
    	# Take step
        self.optim.step()
        self.steps += 1
        
        return 0 # dummy return to save GPU transfer time!
        '''!!!''' 

