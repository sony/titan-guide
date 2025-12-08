import torch
import os
import scipy
from PIL import Image
from diffusers import AudioDiffusionPipeline
import numpy as np
# from .utils import load_audio_dataset, check_grad_fn, rescale_grad
import torchvision
import torchvision.transforms as transforms
mean = [0.585, 0.506, 0.506]
# mean = [0.485, 0.426, 0.436]
# mean = [0.485, 0.456, 0.406]
std= [0.229, 0.224, 0.225]
transform = transforms.Compose([transforms.Normalize(mean, std)])
class VideoInpaintingGuidanceNoGrad:

    def __init__(self, args):
        print("VideoInpaintingGuidance TASK>>>>>>>>>>>>>>>")

        self.args = args

        self.device = args.device

    def prepare_target(self, video_path, device):
 
        frames, _, _ = torchvision.io.read_video(str(video_path), output_format="TCHW")
        frames = transform((frames/255.).float()).to(device) 
 
        self.target_video = frames

        return frames

   
    def get_guidance(self, x_need_grad, func=lambda x: x, post_process=lambda x:x, return_logp=False, check_grad=True, **kwargs):

 
        if func is not None:
            x_need_grad = func(x_need_grad)

        if post_process is not None:
            x = post_process(x_need_grad)
        else:
            x = x_need_grad

        mask = torch.ones_like(x) 
        start_idx = 0
        end_idx = x.shape[0]

        mask[start_idx+1:end_idx-1]

        target_videos = self.target_video[:end_idx] * mask

        difference = target_videos - x * mask

        # classifier returns log prob!
        log_probs = -torch.norm(difference, p=2, dim=(1, 2, 3))

        if return_logp:
            return log_probs.sum()

        grad = torch.autograd.grad(log_probs.sum(), x_need_grad)[0]

        return grad 
        
    

if __name__ == "__main__":
    args = None
    guide = VideoInpaintingGuidanceNoGrad(args)

    vide = guide.prepare_target_video('/mnt/data2/chris/datasets/vggsound/reference_videos_2secs/aAkFRt2rZIg_000130.mp4')
    print(">>>vide: " ,vide.shape)