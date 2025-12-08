import torch

from .networks.style_CLIP import StyleCLIP
from .utils import check_grad_fn, rescale_grad, ban_requires_grad

class StyleTransferGuidance:

    def __init__(self, guide_network, target, device):
        
        # e.g., 'openai/clip-vit-base-patch16'
        self.guide_network = guide_network
        
        # an image path
        self.target = target

        print(">>> guide networ, target:", guide_network, target)
        
        self.device = device
        self._load_model()

        # self.
    
    def _load_model(self):
        self.model = StyleCLIP(self.guide_network, self.device, target=self.target)

        ban_requires_grad(self.model)

    # @torch.enable_grad()
    def get_guidance(self, x_need_grad, func=lambda x:x, post_process=lambda x:x, return_logp=False, check_grad=True, **kwargs):
        with torch.cuda.amp.autocast(enabled=True):
            self.model.grad = None
            # if check_grad:
            #     check_grad_fn(x_need_grad)
            
            if func is not None:
                x_need_grad = func(x_need_grad)

            ### divide into two:
            len_frame = x_need_grad.shape[2]
            step_length = len_frame//2
            
            if post_process is not None:
                x = post_process(x_need_grad)
            else:
                x =  x_need_grad.squeeze(0)
            

            if len(x.shape) == 5:
                x = x.squeeze().transpose(0, 1) 
            log_probs = self.model(x)
 
            if log_probs.shape[0] > 1:
                log_probs = log_probs.mean()
 
            if return_logp:
                return log_probs
 
            grad = torch.autograd.grad(log_probs*1, x_need_grad)[0] ### STYLE transfoer: *1000 loss and guidance_strength=1000

 

            return grad  
 