from .base import BaseGuidance
from diffusers.utils.torch_utils import randn_tensor
from torch.cuda.amp import autocast
import gc
import torch

## reference: https://github.com/vvictoryuki/FreeDoM/blob/1394b1dc5807fb01db6a26d2dc42ca05e3d2eaf5/CN/cldm/ddim_hacked.py#L451

class FreedomGuidance(BaseGuidance):

    def __init__(self, args, **kwargs):
        super(FreedomGuidance, self).__init__(args, **kwargs)

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
        t: int,
        unet: torch.nn.Module,
        ts: torch.LongTensor,
        alpha_prod_ts: torch.Tensor,
        alpha_prod_t_prevs: torch.Tensor,
        eta: float,
        **kwargs,
    ) -> torch.Tensor:
        
        rho = self.get_rho(t, alpha_prod_ts, alpha_prod_t_prevs)
        
        alpha_prod_t = alpha_prod_ts[t]
        alpha_prod_t_prev = alpha_prod_t_prevs[t]
        t = ts[t]

        text_embed = kwargs.get("text_embed")

        with torch.cuda.amp.autocast(enabled=True):
            for recur_step in range(1):#range(self.args.recur_steps):
                
                # we follow the exact algorithm in the paper
                # with torch.no_grad():
                unet.grad=None
                self.guider.vae.grad=None
                self.vae = None
                torch.cuda.empty_cache()
                # line 4 ~ 5

                # print(">>> xx: ", x.shape, t)
                # print(">>>>> cuda memory freedom before1: ", torch.cuda.mem_get_info(device="cuda:0"), torch.cuda.memory_allocated(0))
                # unet.grad=None
                # self.guider.vae.grad=None
                with torch.no_grad():
                    epsilon = unet(x, t, text_embed)
                    x_prev = self._predict_x_prev_from_eps(x, epsilon, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs)

                # unet.grad=None
                # self.guider.vae.grad=None
                # torch.cuda.empty_cache()
                # gc.collect()

                # line 6
                func = unet #lambda zt: (zt - (1 - alpha_prod_t) ** (0.5) * unet(zt, t, text_embed=text_embed)) / (alpha_prod_t ** (0.5))

               
                kwargs["method"] = "seeandhear"
                kwargs["t"] = t
                kwargs["alpha_prod_t"] = alpha_prod_t
                kwargs["text_embed"] = text_embed
                guidance = self.guider.get_guidance(x.detach().requires_grad_(), func, **kwargs).detach().clone()
                
                x_prev = x_prev  + rho * guidance.detach() #rho * guidance
  
        print(x.abs().max().item())
        outputs = {}
        outputs["latent"] = x_prev
        outputs["text_embeds"] = text_embed
        return outputs
        # return x_prev