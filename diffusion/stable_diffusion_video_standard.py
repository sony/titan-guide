import math
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
from tqdm import tqdm
from typing import List

from diffusers.utils.torch_utils import randn_tensor

from utils.configs import Arguments
from .base import BaseSampler
from methods.base import BaseGuidance
# from diffusers import StableDiffusionPipeline
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter, UNet2DConditionModel, UNetMotionModel,AutoencoderKL
from  diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from utils.env_utils import *
from diffusers.utils import export_to_gif, export_to_video
from torch.cuda.amp import autocast
import random
import numpy as np
import os

class StableDiffusionVideoStdSampler(BaseSampler):

    def __init__(self, args: Arguments):

        super(StableDiffusionVideoStdSampler, self).__init__(args)
        # self.video_size = args.video_size
        self.inference_steps = args.inference_steps
        self.eta = args.eta
        self.log_traj = args.log_traj
        self.generator = torch.manual_seed(self.seed)

        self.save_path = args.save_path

        random.seed(self.seed)
        np.random.seed(self.seed)
        # torch.manual_seed(seed)
        torch.cuda.manual_seed_all(self.seed)

        # FIXME: need to send batch_id to guider
        self.args = args
        # prepare unet, prev_t, alpha_prod, alpha_prod_prev...
        self._build_diffusion(args)

    @torch.no_grad()
    def decode(self, latents):
        return self.sd_pipeline.vae.decode(latents / self.sd_pipeline.vae.config.scaling_factor, return_dict=False, generator=self.generator)[0]

    @torch.no_grad()
    def decode_latents(self, latents, decode_chunk_size: int = 16):
        # latents = 1 / self.sd_pipeline.vae.config.scaling_factor * latents
        with autocast(True):
            # print(">>> latenttss: ", latents.shape) ### torch.Size([1, 4, 15, 32, 32])
            batch_size, channels, num_frames, height, width = latents.shape
            latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

            video = []
          
            video = self.sd_pipeline.vae.decode(latents / self.sd_pipeline.vae.config.scaling_factor, return_dict=False, generator=self.generator)[0]
 
            video = video[None, :].reshape((batch_size, num_frames, -1) + video.shape[2:]).permute(0, 2, 1, 3, 4)
            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
            video = video.float()
            return video

    @torch.no_grad()
    def _build_diffusion(self, args):
        
        '''
            Different diffusion models should be registered here
        '''
        # self.sd_pipeline = StableDiffusionPipeline.from_pretrained(args.model_name_or_path).to(self.device)
        # self.sd_pipeline = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16) #StableDiffusionPipeline.from_pretrained(args.model_name_or_path).to(self.device)
        adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float32)
        self.sd_pipeline = AnimateDiffPipeline.from_pretrained(args.model_name_or_path, motion_adapter=adapter, torch_dtype=torch.float32).to(self.device)
        
        self.sd_pipeline.scheduler.set_timesteps(args.inference_steps)
        self.scheduler = self.sd_pipeline.scheduler

        self.video_processor = self.sd_pipeline.video_processor

        unet_cond = UNet2DConditionModel.from_pretrained(args.model_name_or_path, subfolder="unet", torch_dtype=torch.float32).to(self.device)
        unet = UNetMotionModel.from_unet2d(unet_cond, adapter).to(self.device)
        
        # unet = self.sd_pipeline.unet
        unet.eval()

        for param in unet.parameters():
            param.requires_grad = False

        
        # self.scheduler.set_timesteps(args.inference_steps)
        ts = self.scheduler.timesteps
        print(">>> ts: ", len(ts))

        alpha_prod_ts = self.scheduler.alphas_cumprod[ts]
        alpha_prod_t_prevs = torch.cat([alpha_prod_ts[1:], torch.ones(1) * self.scheduler.final_alpha_cumprod])

        print(">> self.sd_pipeline.unet.config.sample_size: ", self.sd_pipeline.unet.config.sample_size)
        self.height = self.width = self.sd_pipeline.unet.config.sample_size * self.sd_pipeline.vae_scale_factor

        self.height = 256 #336 #self.height //2 #//4 *3 #//2 #//4 *3
        self.width = 256 #336 #self.width //2 #//4 *3 #//2 #//4 *3
        self.num_frames = 16 #32 #16

        # prepare prompts: str or List[str]
        self.prompts = self._prepare_prompts(args)  

        # FIXME: classifier-free guidance params
        self.do_classifier_free_guidance = True
        self.guidance_scale = 7.5

        # check inputs. Raise error if not correct
        self.sd_pipeline.check_inputs(self.prompts, self.height, self.width, callback_steps=None)

        self.unet, self.ts, self.alpha_prod_ts, self.alpha_prod_t_prevs = unet, ts, alpha_prod_ts, alpha_prod_t_prevs

    
    def _prepare_prompts(self, args):
        self.ids = None
        if args.dataset == 'parti_prompts':
            prompts = [line.strip() for line in open(PARTIPROMPOTS_PATH, 'r').readlines()][:args.num_samples]
        elif 'vgg' in args.dataset:
            dataset_files = open(args.dataset, 'r').readlines()
            prompts = [line.strip().split(",")[1].replace("\"","") for line in dataset_files][:args.num_samples]
            self.ids = [line.strip().split(",")[0] for line in dataset_files][:args.num_samples]
        else:
            prompts = ["flower"] * args.num_samples
        
        return prompts

    @torch.no_grad()
    def sample(self, sample_size: int, guidance: BaseGuidance):
        
        tot_samples = []
        n_batchs = math.ceil(sample_size / self.per_sample_batch_size)

 

        for batch_id in range(n_batchs):
            
            self.args.batch_id = batch_id

            prompts = self.prompts[batch_id * self.per_sample_batch_size: min((batch_id + 1) * self.per_sample_batch_size, len(self.prompts))]
            print(">>> prompts: ", prompts)
            if self.ids is not None:
                ids = self.ids[batch_id * self.per_sample_batch_size: min((batch_id + 1) * self.per_sample_batch_size, len(self.prompts))]
                self.text_id = ids[0][:-2]#.replace("_0.",".")
            else:
                self.text_id = prompts
   
            
            if isinstance(self.text_id, list):
                text_id = self.text_id[0].replace(".","").replace(",","").replace("|","").replace("\"","").replace("'","")
            else:
                text_id = self.text_id.replace(".","").replace(",","").replace("|","").replace("\"","").replace("'","")
 
            prompt_embeds, negative_prompt_embeds = self.sd_pipeline.encode_prompt(
                prompts, 
                self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=[""]#["bad quality, worse quality, cartoon"]
            )

            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            
            prompt_embeds = prompt_embeds.repeat_interleave(repeats=self.num_frames, dim=0)

            latents = self.sd_pipeline.prepare_latents(
                len(prompts),
                self.sd_pipeline.unet.config.in_channels,
                self.num_frames,
                self.height,
                self.width,
                prompt_embeds.dtype,
                self.device,
                generator=self.generator
            )
             ### SAKTI

            # self.sd_pipeline = None
            if 'imagebind' in self.args.task :#hasattr(guidance.guider, 'prepare_target'): ### means it is imagebind guidance
                AUDIOPATH='/mnt/data2/dataset/vggsound/svg_no/audio2video/audio'
                target = {'audio': f'{AUDIOPATH}/{self.text_id}.wav', 'text': prompts}
                # print("???>>> guidance: ", guidance.guider)
                guidance.guider.prepare_target(target, device=self.device)
            elif 'video_inpainting' in self.args.task :#hasattr(guidance.guider, 'prepare_target_video'): ### means it is imagebind guidance
                VIDEOPATH=self.args.target#'/mnt/data2/dataset/vggsound/svg_no/audio2video/audio'
                target = f'{VIDEOPATH}/{self.text_id}.mp4'#{'video': f'{AUDIOPATH}/{self.text_id}.mp4', 'text': prompts}
       
                with torch.no_grad():
                    frame_target = guidance.guider.prepare_target(target, device=self.device)
                    encode_vid = self.sd_pipeline.vae.encode(frame_target, return_dict=False)[0].sample() * self.sd_pipeline.vae.config.scaling_factor 
                    encode_vid = encode_vid.transpose(0, 1).unsqueeze(0) ###  torch.Size([16, 4, 32, 32])
                    start_idx = 0
                    end_idx = latents.shape[2]
     
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            
            tot_samples = [] 
            test_task = not "nograd" in self.args.task 
            with torch.cuda.amp.autocast(enabled=test_task):
                for t in tqdm(range(self.inference_steps), total=self.inference_steps):
         
                    if 'video_inpainting' in self.args.task : ## only for titan-guide gradest
                        latents[:, :, :start_idx+2] = encode_vid[:, :, :start_idx+2]
                        latents[:, :, end_idx-2:] = encode_vid[:, :, end_idx-2:]

                    def stable_diffusion_unet(latents, t, text_embed=None):
                        # with autocast(True):
                        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        noise_pred = self.unet(latent_model_input.float(), t.float(), encoder_hidden_states=prompt_embeds.float())[0]

                        # perform guidance
                        if self.do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                        
                        return noise_pred

                
                    kwargs = { 'looplength': 1}
                     
                    if "titan" in self.args.guidance_name:
                        # torch.backends.cuda.enable_mem_efficient_sdp(False)
                        latents = guidance.guide_step(
                            latents, t, stable_diffusion_unet, self.sd_pipeline.vae,
                            self.ts,
                            self.alpha_prod_ts, 
                            self.alpha_prod_t_prevs,
                            self.eta,
                            **kwargs
                        )
                    else:
                          latents = guidance.guide_step(
                            latents, t, stable_diffusion_unet, #self.sd_pipeline.vae,
                            self.ts,
                            self.alpha_prod_ts, 
                            self.alpha_prod_t_prevs,
                            self.eta,
                            **kwargs
                        )
 
            image = self.decode_latents(latents) #self.decode(latents)

            tot_samples.append(image.clone().cpu())
            tot_samples = torch.concat(tot_samples)
            videos = self.sd_pipeline.video_processor.postprocess_video(video=tot_samples, output_type="pil")

            print(">>> save at: ", f"{self.save_path}/{text_id}.mp4")
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            export_to_video(videos[0], f"{self.save_path}/{text_id}.mp4", fps=8)
            
            # export_to_video(videos[0], f"/mnt/data2/chris/outputs/vgg/test_ours_epsdiff/{self.text_id}.mp4", fps=8)
        
        
        return torch.concat(tot_samples)
        
    def tensor_to_obj(self, x):

        # print(">>> xx:: ", x.shape)
        
        videos = self.sd_pipeline.video_processor.postprocess_video(video=x, output_type="pil")
        export_to_video(videos[0], f"/mnt/data2/chris/outputs/vgg/test_seeingandhearing_textupd/{self.text_id}.mp4", fps=8)
        
        return videos
    
    def obj_to_tensor(self, objs: List[Image.Image]) -> torch.Tensor:
        '''
            convert a list of PIL images into tensors
        '''
        images = [to_tensor(pil_image) for pil_image in objs]
        tensor_images = torch.stack(images).to(self.device)
        return tensor_images * 2 - 1
