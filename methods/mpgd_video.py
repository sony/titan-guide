from .base import BaseGuidance
from diffusers.utils.torch_utils import randn_tensor

import torch

class MPGDGuidanceVideo(BaseGuidance):

    def __init__(self, args, **kwargs):
        super(MPGDGuidanceVideo, self).__init__(args, **kwargs)
    
    def guide_step(
        self,
        x: torch.Tensor,
        t: int,
        unet: torch.nn.Module,
        ts: torch.LongTensor,
        alpha_prod_ts: torch.Tensor,
        alpha_prod_t_prevs: torch.Tensor,
        eta: float,
        **kwargs,
    ) -> torch.Tensor:
        """
            This function implements Algorithm 1 in MPGD paper.
        """

        alpha_prod_t = alpha_prod_ts[t]
        alpha_prod_t_prev = alpha_prod_t_prevs[t]
        t = ts[t]

        eps = unet(x, t)

        # predict inital x0 using xt and epsilon
        x0 = self._predict_x0(x, eps, alpha_prod_t, **kwargs)

      
        len_frame = x0.shape[2]
        with torch.cuda.amp.autocast(enabled=True) and torch.enable_grad():
            guidance0 = self.guider.get_guidance(x0[:, :, :len_frame//2 ].clone().detach().requires_grad_(),  **kwargs)
            guidance1 = self.guider.get_guidance(x0[:, :, len_frame//2:].clone().detach().requires_grad_(),   **kwargs)
            
            guidance = torch.cat([guidance0, guidance1], dim=2).clone().detach()

        guide_score = guidance #self.guider.get_guidance(x0.clone().detach().requires_grad_(), **kwargs)
        x0 += self.args.guidance_strength * guide_score  

        # update xt using x0 and call DDIM sample prediction
        xt = (1 - alpha_prod_t) ** (0.5) * eps + alpha_prod_t ** (0.5) * x0

        x_prev = self._predict_x_prev_from_zero(
            xt, x0, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs
        )
        
        return x_prev