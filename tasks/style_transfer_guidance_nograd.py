import torch

from .networks.style_CLIP import StyleCLIP
from .utils import check_grad_fn, rescale_grad, ban_requires_grad

class StyleTransferGuidanceNoGrad:

    def __init__(self, guide_network, target, device):
        
        # e.g., 'openai/clip-vit-base-patch16'
        self.guide_network = guide_network
        
        # an image path
        self.target = target
        
        self.device = device
        self._load_model()

        # self.
    
    def _load_model(self):
        self.model = StyleCLIP(self.guide_network, self.device, target=self.target)

        ban_requires_grad(self.model)

 
    def get_guidance(self, x_need_grad, func=lambda x:x, post_process=lambda x:x, return_logp=False, 
                     check_grad=False, get_video_grad=False, grad_in_inputspace=False, 
                     calc_mean=True, **kwargs):
 
        x = None
 
        if get_video_grad: ## gradient sampling
            with torch.enable_grad():
                if func is not None:
                    x_need_grad = x_need_grad.requires_grad_(True)
                    # print(">>> x_need_grad: ", x_need_grad.shape) ## torch.Size([1, 4, 16, 32, 32])
                    x = func(x_need_grad)
 
                if x is None:
                    x = x_need_grad
                 
                log_probs = self.model(x)
 
                if log_probs.shape[0] > 1 and calc_mean:
                    log_probs = log_probs.mean()
   
                if return_logp:
                    return log_probs#*100
                
 
                grad = torch.autograd.grad(log_probs*1, x_need_grad)[0]

        else: ## feed forward
            if check_grad:
                # 
                if post_process is not None:
                    x = post_process(x_need_grad)
                else:
                    x = x_need_grad
 
                log_probs = self.model(x)

 
                if log_probs.shape[0] > 1  and calc_mean:
                    log_probs = log_probs.mean()
 
                if return_logp:
                    return log_probs#*100
            else:
                    
                with torch.enable_grad():
                    if post_process is not None:
                        x = post_process(x_need_grad)
                    else:
                        x = x_need_grad
                    
                    if grad_in_inputspace:
                        x = x.requires_grad_(True)
 
                    log_probs = self.model(x)
 
                    if log_probs.shape[0] > 1:
                        log_probs = log_probs.mean()
 
                    if return_logp:
                        return log_probs#*100
 
                    if grad_in_inputspace:
                        grad = torch.autograd.grad(log_probs*1, x)[0]
                        grad = grad.unsqueeze(0)
                        grad = torch.nn.functional.interpolate(grad, size=(4, 48, 48) )[0]
                    else:
                        grad = torch.autograd.grad(log_probs*1, x_need_grad)[0]
 

        return rescale_grad(grad, clip_scale=1.0, **kwargs)