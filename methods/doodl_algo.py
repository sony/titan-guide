from .base import BaseGuidance
from diffusers.utils.torch_utils import randn_tensor
import torch.func as fc
from functools import partial

import torch
# from .doodl.helper_functions import *
# from .doodl.memcnn import InvertibleModuleWrapper

from torch import nn

class MixingLayer(nn.Module):
    """
    This does the mixing layer of EDICT 
    https://arxiv.org/abs/2211.12446
    Equations 12/13
    """
    
    def __init__(self, mix_weight=0.93):
        super(MixingLayer, self).__init__()
        self.p = mix_weight
        
    def forward(self, input_x):
        input_x0, input_x1 = input_x[:1], input_x[1:]
        x0 = self.p*input_x0 + (1-self.p)*input_x1
        x1 = (1-self.p)*x0 + self.p*input_x1
        return torch.cat([x0, x1])
    
    def inverse(self, input_x):
        input_x0, input_x1 = input_x.split(1)
        x1 = (input_x1 - (1-self.p)*input_x0) / self.p
        x0 = (input_x0 - (1-self.p)*x1) / self.p
        return torch.cat([x0, x1])

class DOODLGuidance(BaseGuidance):

    def __init__(self, args, **kwargs):
        super(DOODLGuidance, self).__init__(args, **kwargs)
        self.device = args.device
      

    def get_rho(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.rho_schedule == 'decrease':    # beta_t
            scheduler = 1 - alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.rho_schedule == 'increase':  # alpha_t
            scheduler = alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.rho_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)

        return self.args.guidance_strength * scheduler[t] * len(scheduler) / scheduler.sum()
    
    def guide_step(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
    
        guidance = self.guider.get_guidance(x, **kwargs)#.detach().clone()

        return guidance
 