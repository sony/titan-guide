import torch
import torch.nn as nn
import os
from PIL import Image
from torchvision import transforms
from datasets import load_from_disk, load_dataset
from diffusers import StableDiffusionPipeline, AnimateDiffPipeline, MotionAdapter, AutoencoderKL

from functools import partial
import logger

from .style_transfer_guidance import StyleTransferGuidance
from .style_transfer_guidance_nograd import StyleTransferGuidanceNoGrad
from .video_inpainting import VideoInpaintingGuidance
from .video_inpainting_nograd import VideoInpaintingGuidanceNoGrad
# from torch.cuda.amp import autocast
from diffusion.cogvid_pipeline.autoencoder_magvit import AutoencoderKLCogVideoX

class BaseGuider:

    def __init__(self, args):
        self.args = args
        self.generator = torch.manual_seed(args.seed)
        
        self.load_processor()   # e.g., vae for latent diffusion
        self.load_guider()      # guidance network

    def load_processor(self):
        if self.args.data_type == 'text2image':
            sd = StableDiffusionPipeline.from_pretrained(self.args.model_name_or_path)
            self.vae = sd.vae
            self.vae.eval()
            self.vae.to(self.args.device)
            for param in self.vae.parameters():
                param.requires_grad = False
            self.processor = lambda x: self.vae.decode(x / self.vae.config.scaling_factor, return_dict=False, generator=self.generator)[0]
        elif 'text2videocogvid' in self.args.data_type:
            self.processor = lambda x: x 
        elif 'text2videonormal' in self.args.data_type: ### EDIT
           
            vae = AutoencoderKL.from_pretrained(self.args.model_name_or_path, subfolder="vae", torch_dtype=torch.float16).to(self.args.device) #sd = AnimateDiffPipeline.from_pretrained(self.args.model_name_or_path, motion_adapter=adapter, torch_dtype=torch.float16)
            self.vae = vae#.to(self.args.device)
            self.vae.eval()
            # self.vae = self.vae
            for param in self.vae.parameters():
                param.requires_grad = False
            def decoding(input):
                with torch.cuda.amp.autocast(enabled=True):
                    self.vae.grad=None
                    segment_idx=input["segment_idx"]
                    segment_size= input["segment_size"]
                    x = input["x"]
                    temp = (self.vae.decode(x.permute(0, 2, 1, 3, 4).reshape(-1, x.shape[1], *x.shape[3:]) / self.vae.config.scaling_factor, return_dict=False, generator=self.generator)[0]).unsqueeze(0)
                    return temp.squeeze()

            self.processor = lambda x: decoding(x)
        elif 'text2videostd' in self.args.data_type and not "titan" in self.args.guidance_name: ### EDIT
            
            test_task = not "nograd" in self.args.task
            dtype = torch.float32 if not test_task else torch.float16
 
            with torch.cuda.amp.autocast(enabled=test_task):
                adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=dtype)
                
                sd = AnimateDiffPipeline.from_pretrained(self.args.model_name_or_path, motion_adapter=adapter, torch_dtype=dtype)
                self.vae = sd.vae
 
                self.vae.eval()
                self.vae.to(dtype=dtype, device=self.args.device)
                for param in self.vae.parameters():
                    param.requires_grad = False

                def decoding(x):
                    self.vae.grad=None
                    inputs = x.permute(0, 2, 1, 3, 4).reshape(-1, x.shape[1], *x.shape[3:])  / self.vae.config.scaling_factor
                    temp = (self.vae.decode(inputs, return_dict=False, generator=self.generator)[0]).unsqueeze(0)
                    return temp.squeeze()

            self.processor = lambda x: decoding(x)
        
        elif 'text2video' in self.args.data_type: ### EDIT
            self.processor = lambda x: x 
        else:
            self.processor = lambda x: x

    @torch.enable_grad()
    def process(self, x):
        return self.processor(x)

    @torch.no_grad()
    def load_guider(self):
        
        self.get_guidance = None

        # for combined guidance
        device = self.args.device

        guiders = []

        for task, guide_network, target in zip(self.args.tasks, self.args.guide_networks, self.args.targets):

            if task == 'style_transfer':
                guider = StyleTransferGuidance(guide_network, target, device)
            elif task == 'style_transfer_nograd':
                guider = StyleTransferGuidanceNoGrad(guide_network, target, device)
            elif task == 'video_inpainting_nograd':
                guider = VideoInpaintingGuidanceNoGrad(self.args)
            elif task == 'video_inpainting':
                guider = VideoInpaintingGuidance(self.args)
            else:
                raise NotImplementedError
            
            guiders.append(guider)
        
        if len(guiders) == 1:
            self.get_guidance = partial(guider.get_guidance, post_process=self.process) ### EDIT to add post process e.g., decoding
            if hasattr(guider, 'prepare_target'):
                self.prepare_target = guider.prepare_target
            self.guider = guider
        else:
            self.get_guidance = partial(self._get_combined_guidance, guiders=guiders)

        if self.get_guidance is None:
            raise ValueError(f"Unknown guider: {self.args.guider}")
    
    def _get_combined_guidance(self, x, guiders, *args, **kwargs):
        values = []
        for guider in guiders:
            values.append(guider.get_guidance(x, post_process=self.process, *args, **kwargs))
        return sum(values)