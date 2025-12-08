from .base import BaseGuidance
from diffusers.utils.torch_utils import randn_tensor

import math
from torch.autograd import grad
import torch
from functools import partial

from tasks.utils import rescale_grad


class TFGGuidanceVideo(BaseGuidance):

    def __init__(self, args, **kwargs):
        super(TFGGuidanceVideo, self).__init__(args, **kwargs)
        self.device = args.device

    @torch.enable_grad()
    def tilde_get_guidance(self, x0, mc_eps, vae, return_logp=False, **kwargs):

 

        def decoding(x):
            
            inputs = x.permute(0, 2, 1, 3, 4).reshape(-1, x.shape[1], *x.shape[3:])  / vae.config.scaling_factor
 
            if kwargs.get("cogvid") is not None:
                temp = (vae.decode(x.permute(0, 2, 1, 3, 4) / vae.config.scaling_factor, return_dict=False)[0]).unsqueeze(0) #(vae.decode(inputs, return_dict=False)[0])
            else:
                #### animatediff
                temp = (vae.decode(inputs, return_dict=False, generator=self.generator)[0]).unsqueeze(0)

            return temp.squeeze()
        
        flat_x0 = (x0[None] + mc_eps).reshape(-1, *x0.shape[1:])
 
        outs = self.guider.get_guidance(flat_x0, return_logp=True, check_grad=False, post_process=decoding, **kwargs)
 
        avg_logprobs = torch.logsumexp(outs.reshape(mc_eps.shape[0], x0.shape[0]), dim=0) - math.log(mc_eps.shape[0])
 
        
        if return_logp:
            return avg_logprobs

        _grad = torch.autograd.grad(avg_logprobs.sum(), x0)[0]
        _grad = rescale_grad(_grad, clip_scale=self.args.clip_scale, **kwargs)
        return _grad
    
    def get_noise(self, std, shape, eps_bsz=4, **kwargs):
        if std == 0.0:
            return torch.zeros((1, *shape), device=self.device)
        return torch.stack([self.noise_fn(torch.zeros(shape, device=self.device), std, **kwargs) for _ in range(eps_bsz)]) 
 
    def get_rho(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.rho_schedule == 'decrease':    # beta_t
            scheduler = 1 - alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.rho_schedule == 'increase':  # alpha_t
            scheduler = alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.rho_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)

        return self.args.rho * scheduler[t] * len(scheduler) / scheduler.sum()

    def get_mu(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.mu_schedule == 'decrease':    # beta_t
            scheduler = 1 - alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.mu_schedule == 'increase':  # alpha_t
            scheduler = alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.mu_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)

        return self.args.mu *  scheduler[t] * len(scheduler) / scheduler.sum()
    
    def get_std(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.sigma_schedule == 'decrease':    # beta_t
            scheduler = (1 - alpha_prod_ts) ** 0.5
        elif self.args.sigma_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)

        return self.args.sigma *  scheduler[t]

    def _predict_x_prev_from_v_cogvid(
        self,
        xt: torch.Tensor,
        vv: torch.Tensor,
        alpha_prod_t: torch.Tensor,
        alpha_prod_t_prev: torch.Tensor,
        eta: float,
        t: torch.LongTensor,
        **kwargs,
    ) -> torch.Tensor:
        
        '''
            This function predicts x_{t-1} using Equation (12) in DDIM paper.
        '''

        sigma = eta * (
            (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        ) ** (0.5)

 
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
 
        std_dev_t = sigma 
 

        pred_original_sample = (alpha_prod_t**0.5) * xt - ((1-alpha_prod_t)**0.5) * vv
        pred_epsilon = (alpha_prod_t**0.5) * vv + ((1-alpha_prod_t)**0.5) * xt
        

        # return pred_epsilon

        # Equation (12) in DDIM sampling
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if eta > 0 and t.item() > 0:
            prev_sample = self.noise_fn(prev_sample, sigma=std_dev_t, **kwargs)
          
        
        return prev_sample

    def _predict_x_prev_from_zero_v_cogvid(
        self,
        xt: torch.Tensor,
        x0: torch.Tensor,
        alpha_prod_t: torch.Tensor,
        alpha_prod_t_prev: torch.Tensor,
        eta: float,
        t: torch.LongTensor,
        **kwargs,
    ) -> torch.Tensor:
        
        '''
            This function first compute (updated) eps from x_0, and then predicts x_{t-1} using Equation (12) in DDIM paper.
        '''
        
        # new_epsilon = (
        #     (xt - alpha_prod_t ** (0.5) * x0) / (1 - alpha_prod_t) ** (0.5)
        # )
        new_v =  ( xt - (alpha_prod_t**0.5) *x0)/((1-alpha_prod_t)**0.5) #* vvpred_original_sample =

        return self._predict_x_prev_from_v_cogvid(xt, new_v, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs)


    
    def guide_step(
        self,
        x: torch.Tensor,
        t: int,
        unet: torch.nn.Module,
        vae: torch.nn.Module,
        ts: torch.LongTensor,
        alpha_prod_ts: torch.Tensor,
        alpha_prod_t_prevs: torch.Tensor,
        eta: float,
        **kwargs,
    ) -> torch.Tensor:
        alpha_prod_t = alpha_prod_ts[t]
        alpha_prod_t_prev = alpha_prod_t_prevs[t]

        rho = self.get_rho(t, alpha_prod_ts, alpha_prod_t_prevs)
        mu = self.get_mu(t, alpha_prod_ts, alpha_prod_t_prevs)
        std = self.get_std(t, alpha_prod_ts, alpha_prod_t_prevs)

        t = ts[t]   # convert from int space to tensor space

        len_frame = x.shape[1]

        mc_eps_list = []
        num_segment = 4
        segment_size = len_frame//num_segment

        for idx in range(num_segment):
            mc_eps_list.append(self.get_noise(std, x[:, idx*segment_size:(idx+1)*segment_size].shape, self.args.eps_bsz, **kwargs))
            # mc_eps_2 = self.get_noise(std, x[:, :len_frame//2].shape, self.args.eps_bsz, **kwargs)
        mc_eps = torch.cat(mc_eps_list, dim=2) 
        # print(">>> kwargs: ", kwargs)

        for recur_step in range(self.args.recur_steps):

            # sample noise to estimate the \tilde p distribution
            # mc_eps = self.get_noise(std, x.shape, self.args.eps_bsz, **kwargs)

            # Compute guidance on x_t, and obtain Delta_t
            # rho = 0. ## EDITED
            with torch.amp.autocast(device_type="cuda") and torch.enable_grad():
                if rho != 0.0:

                    x_g = x[:,   :  ].clone().detach().requires_grad_()
                    x0 = self._predict_x0(x_g, unet(x_g, t), alpha_prod_t, **kwargs)
                    
                    # print(">>>>XX)000 L: ", x0.shape)
                    # with torch.cuda.amp.autocast(enabled=True) and torch.enable_grad():
                    #     guidance0 = self.guider.get_guidance(x0[:, :, :len_frame//2 ].clone().detach().requires_grad_(),  **kwargs)
                    #     guidance1 = self.guider.get_guidance(x0[:, :, len_frame//2:].clone().detach().requires_grad_(),   **kwargs)

                    ### guidance = torch.cat([guidance0, guidance1], dim=2).clone().detach()
                    if kwargs.get("cogvid") is not None:

                        # tbbt = torch.cuda.get_device_properties(0).total_memory
                        # r = torch.cuda.memory_reserved(0)
                        # a = torch.cuda.memory_allocated(0)
                        # f = r-a
                        # print(">>> memoryyyy t, f, r, a: ", tbbt, f, r, a)
                        len_frame = x.shape[1]

                        Delta_t_list = []
                        for idx in range(num_segment):
                            x_g_idx = x[:, idx*segment_size:(idx+1)*segment_size].clone().detach().requires_grad_()
                            x0_idx = self._predict_x0(x_g_idx, unet(x_g_idx, t), alpha_prod_t, **kwargs)
                            logprobs = self.tilde_get_guidance(x0_idx, mc_eps_list[idx].to(dtype=x0_idx.dtype), return_logp=True, vae=vae,**kwargs)
                            Delta_t = grad(logprobs.sum()*0.00001, x_g_idx)[0] ### scalegrad is only for style transfer
                            # print(">> self.args.clip_scale: ", self.args.clip_scale)
                            Delta_t = rescale_grad(Delta_t, clip_scale=self.args.clip_scale*0.00001, **kwargs).detach()
                            Delta_t_list.append(Delta_t * rho)

                        # ### PART 1
                        
                        # # x_g_1 = x[:,  :  ].clone().detach().requires_grad_()
                        # x_g_1 = x[:,  :len_frame//2  ].clone().detach().requires_grad_()
                        # x0_1 = self._predict_x0(x_g_1, unet(x_g_1, t), alpha_prod_t, **kwargs)#[:, ::len_frame//2]
                        # # x_g_1 = x[:,  :len_frame//2 ].clone().detach().requires_grad_()
                        # # x0_1 = x0[:,  :len_frame//2 ]#self._predict_x0(x_g_1, unet(x_g_1, t), alpha_prod_t, **kwargs)
                        # #self._predict_x0(x_g_2, unet(x_g_2, t), alpha_prod_t, **kwargs)

                        # logprobs = self.tilde_get_guidance(
                        #     x0_1, mc_eps_1.to(dtype=x0_1.dtype), return_logp=True, vae=vae,**kwargs)
                        # Delta_t = grad(logprobs.sum(), x_g_1)[0]
                        # Delta_t = rescale_grad(Delta_t, clip_scale=self.args.clip_scale, **kwargs).detach()
                        # Delta_t_1 = Delta_t * rho

                        # ### PART 2
                        # x_g_2 = x[:,   len_frame//2:  ].clone().detach().requires_grad_()
                        # x0_2 = self._predict_x0(x_g_2, unet(x_g_2, t), alpha_prod_t, **kwargs)#[:,  len_frame//2:  ]
                        # # x_g_2 = x[:,  len_frame//2: ].clone().detach().requires_grad_()
                        # # x0_2 = x0[:,  len_frame//2: ]

                        # # print(">> mc_eps: ", mc_eps.shape, x0_2.shape) ## torch.Size([1, 1, 2, 16, 32, 48]) torch.Size([1, 2, 16, 32, 48])
                        # logprobs = self.tilde_get_guidance(
                        #     x0_2, mc_eps_2.to(dtype=x0_2.dtype), return_logp=True, vae=vae,**kwargs)
                        # Delta_t = grad(logprobs.sum(), x_g_2)[0]
                        # Delta_t = rescale_grad(Delta_t, clip_scale=self.args.clip_scale, **kwargs)
                        # Delta_t_2 = Delta_t * rho
                        # # Delta_t = torch.cat([Delta_t_1, Delta_t_2], dim=2).clone().detach()

                        # x0 = torch.cat([x0_1, x0_2], dim=1)
                        Delta_t = torch.cat(Delta_t_list, dim=1).clone().detach()
                    else:
                        x_g = x.clone().detach().requires_grad_()

                        x0 = self._predict_x0(x_g, unet(x_g, t), alpha_prod_t, **kwargs)

                        len_frame = x0.shape[2]
                        logprobs = self.tilde_get_guidance(
                        x0[:, :, :len_frame//2 ], mc_eps[:, :len_frame//2 ], return_logp=True, vae=vae, **kwargs)
                        Delta_t = grad(logprobs.sum(), x_g)[0]
                        Delta_t = rescale_grad(Delta_t, clip_scale=self.args.clip_scale, **kwargs)
                        Delta_t_1 = Delta_t * rho

                        logprobs = self.tilde_get_guidance(
                            x0[:, :, len_frame//2: ], mc_eps[:, len_frame//2: ], return_logp=True, vae=vae,**kwargs)
                        Delta_t = grad(logprobs.sum(), x_g)[0]
                        Delta_t = rescale_grad(Delta_t, clip_scale=self.args.clip_scale, **kwargs)
                        Delta_t_2 = Delta_t * rho
                        Delta_t = torch.cat([Delta_t_1, Delta_t_2], dim=2).clone().detach()
                    
                else:
                    Delta_t = torch.zeros_like(x)
                    # print(">>> xxxx unet: ", x.shape, x.dtype)
                    x0 = self._predict_x0(x, unet(x, t), alpha_prod_t, **kwargs)
                    ## x_temp = self._predict_x0(x, epsilon, alpha_prod_t, **kwargs)

                # Compute guidance on x_{0|t}
                new_x0 = x0.clone().detach()
                # print(">>> new_x0 unet: ", new_x0.shape, new_x0.dtype) ##  torch.Size([1, 4, 16, 32, 48]) torch.float16
                for _ in range(self.args.iter_steps):
                    if mu != 0.0:
                        if kwargs.get("cogvid") is not None:
                            mul_new_x0_list = []
                            for idx in range(num_segment):
                                mul_new_x0_list.append(self.tilde_get_guidance(new_x0[:, idx*segment_size:(idx+1)*segment_size].detach().requires_grad_(), mc_eps_list[idx].to(dtype=new_x0.dtype), vae=vae, **kwargs).detach())
                                # mul_new_x0_1 = self.tilde_get_guidance(new_x0[:, :len_frame//2].detach().requires_grad_(), mc_eps_1.to(dtype=new_x0.dtype), vae=vae, **kwargs).detach()
                                # mul_new_x0_2 = self.tilde_get_guidance(new_x0[:, len_frame//2:].detach().requires_grad_(), mc_eps_2.to(dtype=new_x0.dtype), vae=vae,**kwargs).detach()
                            mul_new_x0 = torch.cat(mul_new_x0_list, dim=1).clone().detach()
                            new_x0 += mu * mul_new_x0 
                            # self.tilde_get_guidance(
                            #     new_x0.detach().requires_grad_(), mc_eps.to(dtype=new_x0.dtype), vae=vae, **kwargs).detach()
                        else:
                            new_x0 += mu * self.tilde_get_guidance(
                                new_x0.detach().requires_grad_(), mc_eps.to(dtype=new_x0.dtype), vae=vae, **kwargs).detach()
                        


            Delta_0 = new_x0 - x0
            
            # predict x_{t-1} using S(zt, hat_epsilon, t), this is also DDIM sampling
            alpha_t = alpha_prod_t / alpha_prod_t_prev

            if kwargs.get("cogvid") is not None: #kwargs["cogvid"] == 1:
                    # tt = ts[t]
                    # epsilon = unet(x, t)
                    x_prev = self._predict_x_prev_from_zero_v_cogvid(x, x0, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs)
                    # x_prev = self._predict_x_prev_from_v_cogvid(x, epsilon, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs)
            else:
                epsilon = unet(x, t)
                x_prev = self._predict_x_prev_from_eps(x, epsilon, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs)
            
            ### EDIT
            # x_prev = self._predict_x_prev_from_zero(  
            #     x, x0, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs)


            x_prev += Delta_t / alpha_t ** 0.5 + Delta_0 * alpha_prod_t_prev ** 0.5

            x = self._predict_xt(x_prev, alpha_prod_t, alpha_prod_t_prev, **kwargs).detach().requires_grad_(False)
        
        return x_prev
