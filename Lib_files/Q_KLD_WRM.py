import math

import torch
import torch.optim as optimLib
import torch.nn as nn
import torch.nn.functional as F
from utils import AddBias

# TODO: In order to make this code faster:
# 1) Implement _extract_patches as a single cuda kernel
# 2) Compute QR decomposition in a separate process
# 3) Actually make a general KFAC optimizer so it fits PyTorch


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
    
def KFAC_multiplication(current_old_step_scaled, matrix_kronecker_factors, classname):
    m_aa = matrix_kronecker_factors[0];    m_gg = matrix_kronecker_factors [1];
    if classname == 'Conv2d':
        p_grad_mat = current_old_step_scaled.view(current_old_step_scaled.size(0), -1)
    else:
        p_grad_mat = current_old_step_scaled
            
    # Compute KFAC direction in current block based on eigendecomposition
    v = m_gg @ p_grad_mat @ m_aa
    v = v.view(current_old_step_scaled.size()) # store KFAC direction in right format - for current block
    return v

def KFAC_multiplication_using_eigendecom_form(current_old_step_scaled, matrix_kronecker_factors_eigendecomp, classname):
    Q_a = matrix_kronecker_factors_eigendecomp[0];  d_a = matrix_kronecker_factors_eigendecomp[1];
    Q_g = matrix_kronecker_factors_eigendecomp[2]; d_g = matrix_kronecker_factors_eigendecomp[3];
    if classname == 'Conv2d':
        p_grad_mat = current_old_step_scaled.view(current_old_step_scaled.size(0), -1)
    else:
        p_grad_mat = current_old_step_scaled
    
    # Compute KFAC direction in current block based on eigendecomposition
    v1 = Q_g @ p_grad_mat @ Q_a
    v2 = v1 * (d_g.unsqueeze(1) * d_a.unsqueeze(0)) 
    v = Q_g @ v2 @ Q_a.t()
    
    v = v.view(current_old_step_scaled.size()) # store KFAC direction in right format - for current block
    return v

def copy_module_key_dictionary_from_1_to_2(dict1_list, dict2_list, modules):
    for m in modules:
        for dict_in_list1, dict_in_list2 in zip(dict1_list, dict2_list):
            dict_in_list2[m] = dict_in_list1[m] + 0.0

def augment_kronecker_factors(m_aa, m_gg, m_aa_old, m_gg_old, lambdda, rho, modules, step_number):
    m_aa_augmented, m_gg_augmented = {}, {}
    copy_module_key_dictionary_from_1_to_2([m_aa,m_gg],[m_aa_augmented, m_gg_augmented], modules)
    for i, m in enumerate(modules):
        if step_number == 0:
            A_current_k = m_aa[m].data; m_aa_augmented[m] = m_aa_augmented[m] + (1.0/lambdda) * A_current_k
            G_current_k = m_gg[m].data; m_gg_augmented[m] = m_gg_augmented[m] + (1.0/lambdda) * G_current_k
        else:
            A_current_k = (m_aa[m] - rho* m_aa_old[m])/(1.0 - rho); m_aa_augmented[m] = m_aa_augmented[m] + (1.0/lambdda) * A_current_k
            G_current_k = (m_gg[m] - rho* m_gg_old[m])/(1.0 - rho); m_gg_augmented[m] = m_gg_augmented[m] + (1.0/lambdda) * G_current_k
    return m_aa_augmented, m_gg_augmented

class Q_KLD_WRM_Optimizer(optimLib.Optimizer):
    def __init__(self,
                 model,
                 lr_function = lambda epoch_n, iteration_n: 0.1,
                 momentum = 0.9,
                 stat_decay = 0.99,
                 kl_clip = 0.001,
                 damping = 1e-2,
                 weight_decay = 0,
                 fast_cnn = False,
                 Ts = 1,
                 Tf = 1, my_clip_threshold = 1.5):
        defaults = dict()

        def split_bias(module):
            for mname, child in module.named_children():
                if hasattr(child, 'bias'):
                    module._modules[mname] = SplitBias(child)
                else:
                    split_bias(child)

        split_bias(model)

        super(Q_KLD_WRM_Optimizer, self).__init__(model.parameters(), defaults)

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
        self.stat_decay = stat_decay # this is basically rho
        
        # LR stuff
        self.epoch_number = 1
        self.lr_function = lr_function
        self.lr = self.lr_function(self.epoch_number, 0) # self.lambdda = 1.0/lr
        self.prev_lr = self.lr_function(self.epoch_number, 0)
        
        self.kl_clip = kl_clip
        self.damping = damping
        self.weight_decay = weight_decay

        self.fast_cnn = fast_cnn

        self.Ts = Ts # Period: when to accumulate A stateistics, G statistics are currently accumulated all the time
        self.Tf = Tf # Period: When to recompute eigenvalue decomposition
        self.current_iteration_state = 'updating_inverse' # the other 2 possible states are: 'just_after_updating_inverse' and 'long_after_updating_inverse' (2 or more iters after)
        
        self.prev_grad_in_param_list_form = [0 * i.data for i in self.model.parameters()] #set g_{-1} = 0
        self.prev_HAT_grad_in_param_list_form = [0 * i.data for i in self.model.parameters()] # set \hat g_{-1} = 0
        self.prev_step_without_minus_1_over_lbd_in_param_list_form = [0 * i.data for i in self.model.parameters()] # set s_{-1} = 0
        self.m_aa_old = {}; self.m_gg_old = {} # Initialize as None - should be a dictinary with zeros at beginning, but making it None and dealing with it appropriately, for simplicity!

        self.my_clip_threshold = my_clip_threshold # clipping params group independently (my clip)
        
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
        HAT_grad_in_param_list_form = []
        for current_nowadays_grad, current_old_grad, current_old_grad_hat, current_old_step_scaled, m  in zip(grad_at_k_in_param_list_form, 
                                                                                                              self.prev_grad_in_param_list_form, 
                                                                                                              self.prev_HAT_grad_in_param_list_form, 
                                                                                                              self.prev_step_without_minus_1_over_lbd_in_param_list_form, 
                                                                                                              self.modules):
                HAT_grad_k_current = current_nowadays_grad + self.stat_decay * lambda_frac * (current_old_grad_hat - current_old_grad)
                if self.steps == 0: # just set to zero in the right ormat, stored in rhs of next line
                    hat_M_k_minus_1_hat_g_k_minus_1 = current_old_step_scaled
                elif (self.steps == 1) or (self.steps % self.Tf >= 2) or (self.steps % self.Tf == 0 and self.Tf >= 2): 
                    #at k=1, we use \bar F_0 and B_0 which are the same so \hat M_k is a multiple of the identity
                    # but if we compute it the kronecker route it won't be due to the K-FAC approximation, so set it manually
                    # we are also on ths branch if we are 2 steps later or more since we recomputed inverse - same logic about apprx applies
                    # on the inverse renewal step, the previous M_k is still lbd/(lbd + 1)I IF inverse computing period is greater than 2
                    hat_M_k_minus_1_hat_g_k_minus_1 = self.prev_HAT_grad_in_param_list_form[len(HAT_grad_in_param_list_form)] * (1.0)/(1.0 + self.prev_lr) # fraction is lambda/(lambda + 1)
                else: # if we are at 1st step, then just set to zero without overcomplicating
                    matrix_kronecker_factors = [self.m_aa_old[m], self.m_gg_old[m]] # need to takethe old ones...
                    hat_M_k_minus_1_hat_g_k_minus_1 = KFAC_multiplication(current_old_step_scaled, matrix_kronecker_factors, m.__class__.__name__)
                
                HAT_grad_in_param_list_form.append(HAT_grad_k_current - self.stat_decay * lambda_frac * hat_M_k_minus_1_hat_g_k_minus_1)
                
        # set g_{k-1} <- g_k; \hat g_{k-1} <- \hat g_k
        self.prev_grad_in_param_list_form = grad_at_k_in_param_list_form.copy()
        self.prev_HAT_grad_in_param_list_form = HAT_grad_in_param_list_form.copy()
        
        # COPY HAT g_k into the gradient to take step!
        for p, hat_g_k_current in zip(self.model.parameters(), HAT_grad_in_param_list_form):
            p.grad.data.copy_(hat_g_k_current)
        
        # Add weight decay - putting it here means weight decay bit will also be preconditioned by the KFAC matrix
        if self.weight_decay > 0:
            for p in self.model.parameters():
                p.grad.data.add_(self.weight_decay, p.data)

        updates = {}
        
        if self.steps % self.Tf == 0 or self.steps % self.Tf == 1: # we only need to produce these augmented guys when changing the statistics, AND 1 step after that else we're fine using the old ones
            self.m_aa_augmented, self.m_gg_augmented = augment_kronecker_factors(self.m_aa, self.m_gg, self.m_aa_old, self.m_gg_old, 1.0/self.lr, self.stat_decay, self.modules, self.steps)
        
        for i, m in enumerate(self.modules):
            assert len(list(m.parameters())
                       ) == 1, "Can handle only one parameter at the moment"
            classname = m.__class__.__name__
            p = next(m.parameters())

            la = self.damping + self.weight_decay

            if (self.steps % self.Tf == 0) or (self.steps % self.Tf == 1): # when it's time to update the inverse
                # My asynchronous implementation exists, I will add it later.
                # Experimenting with different ways to this in PyTorch.
                                
                self.d_a[m], self.Q_a[m] = torch.symeig(
                    self.m_aa_augmented[m], eigenvectors=True)
                self.d_g[m], self.Q_g[m] = torch.symeig(
                    self.m_gg_augmented[m], eigenvectors=True) # computes the eigen decomposition of bar_A and G matrices

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
            nu = min(1, math.sqrt(self.kl_clip / vg_sum)) # this is the original stuff

        for p in self.model.parameters():
            v = updates[p]; numel_v = torch.numel(v)
            my_clip_factor = min(1, self.my_clip_threshold/(torch.norm(v, p = 2)/math.sqrt(numel_v)))
            #if my_clip_factor < 1: print('CLIPPING ACTIVATED !')
            p.grad.data.copy_(v)
            p.grad.data.mul_(nu * my_clip_factor) # We are multiplying by NU here, which clips the KFAC direction somehow !!!
        
        # obtain KFAC direction
        self.prev_step_without_minus_1_over_lbd_in_param_list_form = grad_at_t_in_param_list_form  = [i.grad.data + 0.0 for i in self.model.parameters()] 
        
        # WRONG, COPIES THE POINTER, HORRIBLE: self.m_aa_old = self.m_aa.copy(); self.m_gg_old = self.m_gg.copy()
        copy_module_key_dictionary_from_1_to_2([self.m_aa, self.m_gg], [self.m_aa_old, self.m_gg_old], self.modules)
        self.prev_lr = self.lr + 0.0
            
        self.optim.step()
        self.steps += 1
        
        return 0 # Dummy return to save CPU time
        '''!!!'''
        

