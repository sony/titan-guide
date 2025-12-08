from .base import BaseGuidance
from diffusers.utils.torch_utils import randn_tensor
import torch.func as fc
from functools import partial
import numpy as np
import torch

def rescale_grad(
    grad: torch.Tensor, clip_scale, **kwargs
):  # [B, N, 3+5]
    node_mask = kwargs.get('node_mask', None)
 
    scale = (grad ** 2).mean(dim=-1)
    if node_mask is not None:  # [B, N, 1]
        scale: torch.Tensor = node_mask.float().squeeze(-1).sum(dim=-1)  # [B]
        clipped_scale = torch.clamp(scale, max=clip_scale)
        co_ef = clipped_scale / scale  # [B]
        grad = grad * co_ef.view(-1, 1, 1)
 
    clipped_scale = torch.clamp(scale, max=clip_scale)
    co_ef = clipped_scale / scale
    return grad * co_ef.view(-1, 1, 1)

class TitanGuideGradEstimate(BaseGuidance):

    def __init__(self, args, **kwargs):
        super(TitanGuideGradEstimate, self).__init__(args, **kwargs)

    def get_rho(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.rho_schedule == 'decrease':    # beta_t
            scheduler = 1 - alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.rho_schedule == 'increase':  # alpha_t
            scheduler = alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.rho_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)
 
        return self.args.guidance_strength * scheduler[t] * len(scheduler) / scheduler.sum()
    
    def advancerun_step(self, 
        x: torch.Tensor,
        t: int,
        unet: torch.nn.Module,
        vae: torch.nn.Module,
        ts: torch.LongTensor,
        alpha_prod_ts: torch.Tensor,
        alpha_prod_t_prevs: torch.Tensor,
        eta: float,
        **kwargs):

        loop_length=5 ## hardcoded first for styleguide only to get the best
        with torch.no_grad():
            
            length = loop_length if t+loop_length < len(alpha_prod_ts) else len(alpha_prod_ts) - t#len(alpha_prod_ts) - t
            print(">> innersteps: ", length)
            for recur_step in range(length):
     
                alpha_prod_t = alpha_prod_ts[t]
                alpha_prod_t_prev = alpha_prod_t_prevs[t]
                tt = ts[t]
                
                # we follow the exact algorithm in the paper
                
                unet.grad=None
                # self.guider.vae.grad=None
                # line 4 ~ 5
 
                epsilon = unet(x, tt)
 

                x_prev = self._predict_x_prev_from_eps(x, epsilon, alpha_prod_t, alpha_prod_t_prev, eta, tt, **kwargs)

                x = x_prev 
                func = lambda zt: (zt - (1 - alpha_prod_t) ** (0.5) * unet(zt, tt)) / (alpha_prod_t ** (0.5))

                t += 1
        
        # # t -= 1
        x = func(x) 
        if kwargs.get("cogvid") is not None:
            videos = (vae.decode(x.permute(0, 2, 1, 3, 4) / vae.config.scaling_factor, return_dict=False)[0])
        else:
             #### animatediff
            videos = (vae.decode(x.permute(0, 2, 1, 3, 4).reshape(-1, x.shape[1], *x.shape[3:]) / vae.config.scaling_factor, return_dict=False, generator=self.generator)[0])#.unsqueeze(0)
             
        loss = self.guider.get_guidance(x_need_grad=videos, return_logp=True, 
                                        check_grad=True, calc_mean=False, **kwargs)
        
  
        return loss

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
 
        rho = self.get_rho(t, alpha_prod_ts, alpha_prod_t_prevs)
        
        alpha_prod_t = alpha_prod_ts[t]
        alpha_prod_t_prev = alpha_prod_t_prevs[t]
        tt = ts[t]
 

        def decoding(x):
            
            inputs = x.permute(0, 2, 1, 3, 4).reshape(-1, x.shape[1], *x.shape[3:])  / vae.config.scaling_factor
 
            # else:
            # temp = (vae.decode(inputs, return_dict=False, generator=self.generator)[0]).unsqueeze(0)
            if kwargs.get("cogvid") is not None:
                temp = (vae.decode(x.permute(0, 2, 1, 3, 4) / vae.config.scaling_factor, return_dict=False)[0]).unsqueeze(0) #(vae.decode(inputs, return_dict=False)[0])
            else:
                #### animatediff
                temp = (vae.decode(inputs, return_dict=False, generator=self.generator)[0]).unsqueeze(0)
                 
            return temp.squeeze()
 

        with torch.no_grad():
            
            for recur_step in range(self.args.recur_steps):
                
                # we follow the exact algorithm in the paper
                
                unet.grad=None
              
                epsilon = unet(x, tt)
 
                latent_noise_direction = torch.randn_like(x) 
                v_params = latent_noise_direction

                if kwargs.get("cogvid") is not None:
                    x_prev = self._predict_x_prev_from_v_cogvid(x, epsilon, alpha_prod_t, alpha_prod_t_prev, eta, tt, **kwargs)
                else:
                    x_prev = self._predict_x_prev_from_eps(x, epsilon, alpha_prod_t, alpha_prod_t_prev, eta, tt, **kwargs)
 
                x_temp = self._predict_x0(x, epsilon, alpha_prod_t, **kwargs)
  

                 # #>>>>>>>>>>>>>>>>>>>>GRAD ESTIMATION per layer>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                with torch.cuda.amp.autocast(enabled=True) and torch.enable_grad():
                    if kwargs.get("cogvid") is  None:
                        factor_size = 4
                        multiplier = x_temp.shape[2]//factor_size
                        idx= 4
                        guidance0 = self.guider.get_guidance(x_temp[:, :, idx:(idx+factor_size) ].clone().detach().requires_grad_(), post_process=decoding,  **kwargs)
     
                        guidance = guidance0.repeat(1, 1, multiplier, 1, 1)
                    else: ## animatediff
                        factor_size = 2
                        idx= 2
                        multiplier = x_temp.shape[1]//factor_size
                        guidance0 = self.guider.get_guidance(x_temp[:, idx:(idx+factor_size) ].clone().detach().requires_grad_(), post_process=decoding,  **kwargs)
                        guidance = guidance0.repeat(1, multiplier, 1, 1, 1)
 
                v_params =  guidance/torch.norm(guidance)
                
                f = partial(
                        self.advancerun_step,
                        t=t,
                        unet=unet,
                        vae=vae,
                        ts=ts,
                        alpha_prod_ts=alpha_prod_ts,
                        alpha_prod_t_prevs=alpha_prod_ts,
                        eta=eta,
                        **kwargs,
                    )
                x_prev = x_prev.requires_grad_(True)

                optim = torch.optim.AdamW([x_prev], lr=1e-1)
                

                optim.zero_grad()
                # Forward AD
 
                loss_im, jvp_im = fc.jvp(f, (x_prev,), (v_params,))
 
                loss = loss_im
                jvp = jvp_im
 
                if jvp.dim()>0 and len(jvp) > 1 and kwargs.get("cogvid") is None:
                    jvp = jvp.reshape(1, 1, jvp.shape[0], 1, 1)

                guide_score = jvp * v_params
                x_prev.grad = -guide_score
                optim.step()
               
                # line 9 ~ 11
                x = self._predict_xt(x_prev, alpha_prod_t, alpha_prod_t_prev, **kwargs)
        
        print(x.abs().max().item())
        return x_prev

