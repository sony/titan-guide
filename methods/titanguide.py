from .base import BaseGuidance
from diffusers.utils.torch_utils import randn_tensor
import torch.func as fc
from functools import partial

import torch

class TitanGuide(BaseGuidance):

    def __init__(self, args, **kwargs):
        super(TitanGuide, self).__init__(args, **kwargs)
        self.generator = torch.manual_seed(args.seed)

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

        loop_length=1
        with torch.no_grad(): 
            length =  loop_length if t+loop_length < len(alpha_prod_ts) else len(alpha_prod_ts) - t#len(alpha_prod_ts) - t
            print(" >> length: ", length)
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

                x = x_prev #self._predict_xt(x_prev, alpha_prod_t, alpha_prod_t_prev, **kwargs)
                func = lambda zt: (zt - (1 - alpha_prod_t) ** (0.5) * unet(zt, tt)) / (alpha_prod_t ** (0.5))

                t += 1
     
        x = func(x)
        
        if kwargs.get("cogvid") is not None:
            videos = (vae.decode(x.permute(0, 2, 1, 3, 4) / vae.config.scaling_factor, return_dict=False)[0])
        else:
             #### animatediff
            videos = (vae.decode(x.permute(0, 2, 1, 3, 4).reshape(-1, x.shape[1], *x.shape[3:]) / vae.config.scaling_factor, return_dict=False, generator=self.generator)[0])#.unsqueeze(0)

        loss = self.guider.get_guidance(x_need_grad=videos, return_logp=True, **kwargs)
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
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = eta * variance ** (0.5)

        pred_original_sample = (alpha_prod_t**0.5) * xt - ((1-alpha_prod_t)**0.5) * vv
        pred_epsilon = (alpha_prod_t**0.5) * vv + ((1-alpha_prod_t)**0.5) * xt

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

        with torch.no_grad():
            print(">> self.args.recur_steps: ", self.args.recur_steps)
            for recur_step in range(1):
                
                # we follow the exact algorithm in the paper
                
                unet.grad=None 
                # line 4 ~ 5
                
                epsilon = unet(x, tt)
 
                if "titanguide" in self.args.guidance_name :
                    latent_noise_direction = torch.randn_like(x) 
                    v_params = latent_noise_direction

                if kwargs.get("cogvid") is not None: #kwargs["cogvid"] == 1:
                    x_prev = self._predict_x_prev_from_v_cogvid(x, epsilon, alpha_prod_t, alpha_prod_t_prev, eta, tt, **kwargs)
                else:
                    x_prev = self._predict_x_prev_from_eps(x, epsilon, alpha_prod_t, alpha_prod_t_prev, eta, tt, **kwargs)

                optim = torch.optim.AdamW([x_prev], lr=8e-2)
                optim.zero_grad()
                # line 6
             
                if kwargs.get("cogvid") is not None:
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
                else:
                    f = partial(
                            self.advancerun_step,
                            t=t+1,
                            unet=unet,
                            vae=vae,
                            ts=ts,
                            alpha_prod_ts=alpha_prod_ts,
                            alpha_prod_t_prevs=alpha_prod_ts,
                            eta=eta,
                            **kwargs, 
                        )

                # Forward AD
                loss_im, jvp_im = fc.jvp(f, (x_prev,), (v_params,))
                loss = loss_im
                jvp = jvp_im
 
                guidance = jvp * v_params 
                x_prev.grad = -guidance
                optim.step()

                # line 9 ~ 11
                x = self._predict_xt(x_prev, alpha_prod_t, alpha_prod_t_prev, **kwargs)
        
        print(x.abs().max().item())
        return x_prev