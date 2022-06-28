import math

import torch
import torch.optim as optimLib
import torch.nn as nn
import torch.nn.functional as F
from utils import AddBias
import numpy as np

#import ipdb

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

def load_params_from_1_to_2(net_from, net_to):
    for p_from, p_to in zip(net_from.parameters(), net_to.parameters()):
        try:
            p_to.data.copy_(p_from.data + 0.0)
        except:
            p_to.data.copy_(p_from.data[:,0] + 0.0) # the bias term is a matrix for p_from and a vector for p_to...
            
class QE_KLD_WRM_Optimizer(optimLib.Optimizer):
    def __init__(self,
                 model, network_generating_function,
                 lr_function = lambda epoch_n, iteration_n: 0.1,
                 momentum=0.9,
                 stat_decay=0.99,
                 kl_clip=0.001,
                 damping=1e-2,
                 weight_decay=0,
                 fast_cnn=False,
                 Ts=1,
                 Tf=1, my_clip_threshold = 1.5,
                 number_inner_SGD_steps = 10, inner_lr_factor = 0.1,
                 force_lr_on_final_step_flag = False,
                 inner_momentum = 0.0, capacity_number_of_prev_nets_stored = 3):
        defaults = dict()

        def split_bias(module):
            for mname, child in module.named_children():
                if hasattr(child, 'bias'):
                    module._modules[mname] = SplitBias(child)
                else:
                    split_bias(child)

        split_bias(model)

        super(QE_KLD_WRM_Optimizer, self).__init__(model.parameters(), defaults)

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

        self.Ts = Ts
        self.Tf = Tf
        
        self.prev_grad_in_param_list_form = [0 * i.data for i in self.model.parameters()] #set g_{-1} = 0
        self.prev_HAT_grad_in_param_list_form = [0 * i.data for i in self.model.parameters()] # set \hat g_{-1} = 0
        self.prev_step_without_minus_1_over_lbd_in_param_list_form = [0 * i.data for i in self.model.parameters()] # set s_{-1} = 0
        self.m_aa_old = {}; self.m_gg_old = {} # Initialize as None - should be a dictinary with zeros at beginning, but making it None and dealing with it appropriately, for simplicity!
        
        #clip and QE subsolver parameters 
        self.my_clip_threshold = my_clip_threshold # clipping params group independently (my clip)
        self.number_inner_SGD_steps = number_inner_SGD_steps
        self.inner_lr_factor = inner_lr_factor
        self.force_lr_on_final_step_flag = force_lr_on_final_step_flag
        self.inner_momentum = inner_momentum
        
        self.optim = optimLib.SGD(
            model.parameters(),
            lr = self.lr * (1 - self.momentum),
            momentum = self.momentum)
        
        # QE_KLD_WRM quantities
        self.capacity_number_of_prev_nets_stored = capacity_number_of_prev_nets_stored
        self.prev_net_list = []
        
        def network_generating_function_to_cuda():
            networkk = network_generating_function() # if not cuda, this is the net generating fct itself
            networkk.to(torch.device('cuda:0'))
            return networkk
        
        self.network_generating_function = network_generating_function_to_cuda
        # save current net in the old net buffer
        current_old_net = self.network_generating_function()
        for temp_p in current_old_net.parameters():
            print('The device is {}'.format(temp_p.device))
        load_params_from_1_to_2(self.model, current_old_net)# current_old_net.load_state_dict(self.model.state_dict())
        self.prev_net_list.append(current_old_net)
        
        self.Currently_doing_QE_part_of_step = False

    def _save_input(self, module, input):
        if self.model.training and (self.steps % self.Ts == 0) and (self.Currently_doing_QE_part_of_step == False):
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
            #print('\nAccumulated FWD statistics at step {}'.format(self.steps))

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
            #print('Accumulated BWD statistics at step {}\n'.format(self.steps))

    def _prepare_model(self):
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                assert not ((classname in ['Linear', 'Conv2d']) and module.bias is not None), \
                                    "You must have a bias as a separate layer"

                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)

    def step(self, epoch_number, data, error_write_path):
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
                try:
                    self.d_a[m], self.Q_a[m] = torch.symeig( # + 0.01* torch.eye(self.m_aa_augmented[m].shape[0], device = torch.device('cuda:0'))
                        self.m_aa_augmented[m] , eigenvectors=True)
                    self.d_g[m], self.Q_g[m] = torch.symeig( # + 0.01* torch.eye(self.m_gg_augmented[m].shape[0], device = torch.device('cuda:0'))
                        self.m_gg_augmented[m] , eigenvectors=True) # computes the eigen decomposition of bar_A and G matrices
                except:
                    np.save(error_write_path + '/m_aa_aug_when_err.npy', self.m_aa_augmented[m].cpu().numpy())
                    np.save(error_write_path + '/m_gg_aug_when_err.npy', self.m_gg_augmented[m].cpu().numpy())
                    np.save(error_write_path + '/m_aa_when_err.npy', self.m_aa[m].cpu().numpy())
                    np.save(error_write_path + '/m_gg_when_err.npy', self.m_gg[m].cpu().numpy())
                    np.save(error_write_path + '/m_aa_old_when_err.npy', self.m_aa_old[m].cpu().numpy())
                    np.save(error_write_path + '/m_gg_old_when_err.npy', self.m_gg_old[m].cpu().numpy())
                    for exception_grad_counter, p in enumerate(self.model.parameters()):
                        np.save(error_write_path + '/grad_{}_when_err.npy'.format(exception_grad_counter), p.grad.cpu().numpy())
                        np.save(error_write_path + '/param_data_{}_when_err.npy'.format(exception_grad_counter), p.data.cpu().numpy())
                    print(self.m_aa[m])
                    raise ValueError('aa^T: Stopped due to failed eigenvalue decomposition')

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
            # if my_clip_factor < 1: print('CLIPPING ACTIVATED !')
            p.grad.data.copy_(v)
            p.grad.data.mul_(nu * my_clip_factor) # We are multiplying by NU here, which clips the KFAC direction somehow !!!
        
        # obtain KFAC direction
        self.prev_step_without_minus_1_over_lbd_in_param_list_form = grad_at_t_in_param_list_form  = [i.grad.data + 0.0 for i in self.model.parameters()] 
        
        # WRONG, COPIES THE POINTER, HORRIBLE: self.m_aa_old = self.m_aa.copy(); self.m_gg_old = self.m_gg.copy()
        copy_module_key_dictionary_from_1_to_2([self.m_aa, self.m_gg], [self.m_aa_old, self.m_gg_old], self.modules)
        self.prev_lr = self.lr + 0.0
        
        '''QE part of the QE step'''
        if len(self.prev_net_list) == self.capacity_number_of_prev_nets_stored: # only take the QE-KLD-WRM step if the capacity is reached, 
            # save current parameters of actual net to have reference and thus get s
            self.Currently_doing_QE_part_of_step = True
            
            current_param_list_for_QE = []
            for p in self.model.parameters():
                current_param_list_for_QE.append(p.data + 0.0) # hpefully .data will return smth that is not tracked by autograd - I think so - at least that's what I need here!
            # take Q-KLD-WRM-STEP - we will further optimize from this (i.e. the Q-KLD-WRM step is the initial guess)
            '''DEBUG!!!'''
            if epoch_number >= 20:
                for idx_counter, p in enumerate(self.model.parameters()):
                    np.save(error_write_path + '/param_just_before_Q_{}_when_err.npy'.format(idx_counter), p.data.cpu().numpy())
                    np.save(error_write_path + '/Q_step_just_before_nan_{}_when_err.npy'.format(idx_counter), p.grad.cpu().numpy())
            '''END DEBUG!!!'''
            self.optim.step()
            '''DEBUG!!!'''
            if epoch_number >= 20:
                for idx_counter, p in enumerate(self.model.parameters()):
                    np.save(error_write_path + '/param_just_before_QE_{}_when_err.npy'.format(idx_counter), p.data.cpu().numpy())
            '''END DEBUG!!!'''
            
            # instantiate INNER-OPTIMIZER - which solves the QE-KLD-WRM subproblem starting from the Q-KLD-WRM step
            # we apply this optimizer directly to the net's parameters - and the QE-KLD-WRM step "loss"
            
            QE_KLD_WRM_inner_optimizer = optimLib.SGD(self.model.parameters(),  lr = self.inner_lr_factor * self.lr,  momentum = self.inner_momentum)
            #QE_KLD_WRM_inner_optimizer = optimLib.Adam(self.model.parameters(), lr = self.inner_lr_factor * self.lr, betas = (0.5,0.55))#
            
            # assemble the loss
            def assemble_QE_KLD_WRM_loss(model, data, prev_net_list, current_inner_step, rho = self.stat_decay, lambbbbda = 1.0/self.lr,):
                our_net_output = model(data) # need to check that self.model and outside model "move" together
                ''' DEBUG '''
                # save nets outputs upon crash
                if self.steps == 2446: # this is the OUTER step counter!
                    print('saving outputs (1)...')
                    #ipdb.set_trace(context = 7)
                    np.save(error_write_path + '/p_net_output_iter_{}.npy'.format(current_inner_step), our_net_output.cpu().detach().numpy())
                ''' END DEBUG '''
                current_net = prev_net_list[0]
                with torch.no_grad(): 
                    current_old_output_target = current_net(data)
                    ''' DEBUG '''
                    if self.steps == 2446:
                        print('saving outputs (2)...')
                        #ipdb.set_trace(context = 7)
                        np.save(error_write_path + '/old_nets_outputs_step_{}_group_{}.npy'.format(current_inner_step, 0), current_old_output_target.cpu().numpy())
                    # note that step "i" would not matter in the above save if we DID NOT HAVE DRPOUT, BUT WE DO HAVE IT!
                    ''' END DEBUG '''
                def exact_classif_sym_KL(no_softmax_p, no_softmax_q): # assume softmax was NOT aplied
                    p = F.softmax(no_softmax_p); q = F.softmax(no_softmax_q)
                    log_p_minus_log_q = F.log_softmax(no_softmax_p) - F.log_softmax(no_softmax_q)
                    p_times_log_p_minus_log_q_SYM = 0.5 * ( p * log_p_minus_log_q + q * ( -1.0 * log_p_minus_log_q) )
                    KL_divs_per_batch_item = torch.sum(p_times_log_p_minus_log_q_SYM, axis = 1)
                    KL_div_of_batch = torch.mean(KL_divs_per_batch_item)
                    #if KL_div_of_batch != 0.5*(F.kl_div(torch.log( p ), q, reduction = 'batchmean') + F.kl_div(torch.log( q ), p, reduction = 'batchmean')):
                    #    raise ValueError('Something went Wrong with the QE KL divergence loss setup')
                    return KL_div_of_batch
                    
                D_KL_loss = exact_classif_sym_KL(our_net_output, current_old_output_target) # her we pass LOGITS
                # Classification (assuming softmax NOT applied): 
                # Regression: torch.mean(torch.sqrt(torch.sum((our_net_output - current_old_output_target)**2, axis = 1)))
                debug_old_out_net_index = 0
                for current_net in prev_net_list[1:] :
                    with torch.no_grad(): 
                        current_old_output_target = current_net(data) # this acts as a target ia  MSE loss
                        ''' DEBUG '''
                        debug_old_out_net_index = debug_old_out_net_index + 1
                        if self.steps == 2446:
                            print('saving outputs (3)...')
                            #ipdb.set_trace(context = 7)
                            np.save(error_write_path + '/old_nets_outputs_step_{}_group_{}.npy'.format(current_inner_step, debug_old_out_net_index), current_old_output_target.cpu().numpy())
                        # note that step "i" would not matter in the above save if we DID NOT HAVE DRPOUT, BUT WE DO HAVE IT!
                        ''' END DEBUG '''
                    D_KL_loss = rho * D_KL_loss + (1 - rho) *  exact_classif_sym_KL(our_net_output, current_old_output_target)
                D_KL_loss = (lambbbbda/2.0) * D_KL_loss # this is multiplying by lambda/2
                return D_KL_loss / 330.0 # /10**5 # this 33.0 is kappa(i) but since it s the same for all i 
                # we can just incorporate it in lambda, so incoporate it in (outer) lr = 1/lambda
            
            # take a few steps of this optimizer - 
            for i in range(self.number_inner_SGD_steps):
                #if (i+1) % (int(self.number_inner_SGD_steps/2)) == 0: print('Inner loop at {}'.format(i))
                self.model.zero_grad()
                QE_KLD_WRM_inner_loss = assemble_QE_KLD_WRM_loss(model = self.model, data = data, prev_net_list = self.prev_net_list, current_inner_step = i) # this is just the D_KL loss part
                QE_KLD_WRM_inner_loss.backward() # the gradient now contains just the D-KL part of te gradient
                
                ''' DEBUG '''
                if self.steps == 2446:
                    print ('saving at step {}'.format(self.steps))
                    for idx_counter, p in enumerate(self.model.parameters()):
                        np.save(error_write_path + '/param_during_QE_step_{}_group_{}.npy'.format(i, idx_counter), p.data.cpu().numpy())
                        np.save(error_write_path + '/KL_grad_during_QE_step_{}_group_{}.npy'.format(i, idx_counter), p.grad.cpu().numpy())
                        np.save(error_write_path + '/KL_loss_during_QE_step_{}_group_{}.npy'.format(i, idx_counter), QE_KLD_WRM_inner_loss.cpu().detach().numpy())
                ''' END DEBUG '''
                
                # we now need to manually add g_k + B_k s, with B_k = \tilde F_KFAC and s = current_theta - saved_theta
                group_index_counter = -1
                for p, g_k_chunk, saved_param_start_pt, m in zip(self.model.parameters(), grad_at_k_in_param_list_form, current_param_list_for_QE, self.modules):
                    group_index_counter += 1
                    delta_theta_chunk = p.data - saved_param_start_pt
                    matrix_kronecker_factors_QE = [(1/self.lr)*(self.m_aa_augmented[m] - self.m_aa[m]) , (1/self.lr)*(self.m_gg_augmented[m] - self.m_gg[m])]
                    B_k_s_chunk = KFAC_multiplication(delta_theta_chunk, matrix_kronecker_factors_QE, m.__class__.__name__)
                    p.grad.data.add_(1.0, g_k_chunk + B_k_s_chunk)
                    
                    if self.steps == 2446:
                        print ('saving(2) at step {}'.format(self.steps))
                        np.save(error_write_path + '/Model_grad_during_QE_step_{}_group_{}.npy'.format(i, group_index_counter), B_k_s_chunk.cpu().numpy())
                    
                    # Clip if norm si too large! - different clips to different params groups!
                    v = p.grad.data + 0.0; numel_v = torch.numel(v)
                    my_clip_factor = min(1, self.my_clip_threshold/(torch.norm(v, p = 2)/math.sqrt(numel_v)))
                    if my_clip_factor != 1: print('CLIPPING ACTIVATED in inner solver! clip_factor = {}'.format(my_clip_factor))
                    p.grad.data.mul_(my_clip_factor)
                    
                QE_KLD_WRM_inner_optimizer.step()
            # that's it - that's the QE-KLD-WRM step!
            self.model.zero_grad()
            # now undo the step to be multiplied by the lr rather than witha  lr of 1
            if self.force_lr_on_final_step_flag == True:
                current_step_norm_squared = 0.0
                for p, prev_params in zip(self.model.parameters(), current_param_list_for_QE):
                    current_step_norm_squared = current_step_norm_squared + torch.norm(p.data - prev_params, p = 2)**2
                for p, prev_params in zip(self.model.parameters(), current_param_list_for_QE):
                    p.data.copy_(prev_params + (self.lr/torch.sqrt(current_step_norm_squared)) * (p.data - prev_params) )
            
            # save current net in old net bufer list - and discard the oldest one
            to_be_deleted_net = self.prev_net_list.pop(0); del to_be_deleted_net
            current_old_net = self.network_generating_function()
            load_params_from_1_to_2(self.model, current_old_net) # current_old_net.load_state_dict(self.model.state_dict())
            self.prev_net_list.append(current_old_net)
            
            self.Currently_doing_QE_part_of_step = False # SWITCH QE part of step flag off at end of step
            
        else: # else take Q-KLD-WRM step
            # take Q-KLD_WRM step
            self.optim.step()
            # save current parameters in the old parameters buffer
            current_old_net = self.network_generating_function()
            load_params_from_1_to_2(self.model, current_old_net) # current_old_net.load_state_dict(self.model.state_dict())
            self.prev_net_list.append(current_old_net)
        self.steps += 1
        
        return 0 # Dummy return for faster computation with GPU
        '''!!!'''

