import math
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
from tqdm import tqdm
from typing import List
import gc
import os

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
import numpy as np
import random

class StableDiffusionVideoNormalSampler(BaseSampler):

    def __init__(self, args: Arguments):

        super(StableDiffusionVideoNormalSampler, self).__init__(args)
        # self.video_size = args.video_size
        self.inference_steps = args.inference_steps
        self.eta = args.eta
        self.log_traj = args.log_traj
        self.generator = torch.manual_seed(self.seed)

        random.seed(self.seed)
        np.random.seed(self.seed)
        # torch.manual_seed(seed)
        torch.cuda.manual_seed_all(self.seed)

        # FIXME: need to send batch_id to guider
        self.args = args
        # prepare unet, prev_t, alpha_prod, alpha_prod_prev...
        self.save_path = args.save_path

        self._build_diffusion(args)

        self.data_type = torch.float16

    @torch.no_grad()
    def decode(self, latents):
        return self.sd_pipeline.vae.decode(latents / self.sd_pipeline.vae.config.scaling_factor, return_dict=False, generator=self.generator)[0]

    @torch.no_grad()
    def decode_latents(self, latents, decode_chunk_size: int = 16):
        # latents = 1 / self.sd_pipeline.vae.config.scaling_factor * latents
        with torch.cuda.amp.autocast(enabled=True):
            # print(">>> latenttss: ", latents.shape) ### torch.Size([1, 4, 15, 32, 32])
            batch_size, channels, num_frames, height, width = latents.shape
            latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

            video = []
            # for i in range(0, latents.shape[0], decode_chunk_size):
            #     batch_latents = latents[i : i + decode_chunk_size]
            #     batch_latents = self.vae.decode(batch_latents).sample
            #     video.append(batch_latents)
            video = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=self.generator)[0]
            # video = torch.cat(video)
            video = video[None, :].reshape((batch_size, num_frames, -1) + video.shape[2:]).permute(0, 2, 1, 3, 4)
            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
            video = video.float()
            return video


    @torch.no_grad()
    def _build_diffusion(self, args):
        
        '''
            Different diffusion models should be registered here
        '''
        self.data_type = torch.float16
        # self.sd_pipeline = StableDiffusionPipeline.from_pretrained(args.model_name_or_path).to(self.device)
        # self.sd_pipeline = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16) #StableDiffusionPipeline.from_pretrained(args.model_name_or_path).to(self.device)
        adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=self.data_type)
        self.sd_pipeline = AnimateDiffPipeline.from_pretrained(args.model_name_or_path, motion_adapter=adapter, torch_dtype=self.data_type).to("cpu") #self.device)
        
        self.sd_pipeline.scheduler.set_timesteps(args.inference_steps)
        self.scheduler = self.sd_pipeline.scheduler

        # self.sd_pipeline.text_encoder = self.sd_pipeline.text_encoder.to(self.device)

        self.video_processor = self.sd_pipeline.video_processor
        self.vae = AutoencoderKL.from_pretrained(args.model_name_or_path, subfolder="vae", torch_dtype=self.data_type).to(self.device)

        unet_cond = UNet2DConditionModel.from_pretrained(args.model_name_or_path, subfolder="unet", torch_dtype=self.data_type).to(self.device)
        unet = UNetMotionModel.from_unet2d(unet_cond, adapter).to(self.device)
        
        # unet = self.sd_pipeline.unet
        unet.eval()

        for param in unet.parameters():
            param.requires_grad = False

        for param in self.vae.parameters():
            param.requires_grad = False

        for param in self.sd_pipeline.text_encoder.parameters():
            param.requires_grad = False
        
        # self.scheduler.set_timesteps(args.inference_steps)
        ts = self.scheduler.timesteps
        print(">>> ts: ", len(ts))

        alpha_prod_ts = self.scheduler.alphas_cumprod[ts]
        alpha_prod_t_prevs = torch.cat([alpha_prod_ts[1:], torch.ones(1) * self.scheduler.final_alpha_cumprod])

        print(">> self.sd_pipeline.unet.config.sample_size: ", self.sd_pipeline.unet.config.sample_size)
        self.height = self.width = self.sd_pipeline.unet.config.sample_size * self.sd_pipeline.vae_scale_factor

        self.height = self.height //2 #//4 *3 #//2 #//4 *3
        self.width = self.width //2 #//4 *3 #//2 #//4 *3
        self.num_frames = 16 #14 #16 #32 #16

        # prepare prompts: str or List[str]
        self.prompts = self._prepare_prompts(args)  

        # FIXME: classifier-free guidance params
        self.do_classifier_free_guidance = True
        self.guidance_scale = 7.5

        # check inputs. Raise error if not correct
        # self.sd_pipeline.check_inputs(self.prompts, self.height, self.width, callback_steps=None)

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
            # prompts = prompts[0][1]

            # if os.path.exists(f"/mnt/data2/chris/outputs/vgg/test_seeingandhearing_textupd/{self.text_id}.mp4"):
            #     continue

            print(">>> self.text_prompt: ", self.text_id)
            # encode input prompts
            # print(">> prompts: ", prompts, self.device)
            with torch.no_grad():
                self.sd_pipeline = self.sd_pipeline.to(self.device)
                prompt_embeds, negative_prompt_embeds = self.sd_pipeline.encode_prompt(
                    prompts, 
                    self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    negative_prompt=["bad quality, worse quality, cartoon"]
                )

                if self.do_classifier_free_guidance:
                    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            
                prompt_embeds = prompt_embeds.repeat_interleave(repeats=self.num_frames, dim=0).detach()

           
            # prompt_embeds.requires_grad_(False)
            # print(">> prompt_embeds.dtype: ", prompt_embeds.dtype)
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
            self.sd_pipeline= self.sd_pipeline.to("cpu")
            # del self.sd_pipeline
            
            # self.sd_pipeline = None
            AUDIOPATH='/mnt/data2/dataset/vggsound/svg_no/audio2video/audio'
            target = {'audio': f'{AUDIOPATH}/{self.text_id}.wav', 'text': prompts}
            # print("???>>> guidance: ", guidance.guider)
            guidance.guider.prepare_target(target, device=self.device)
            tot_samples = [] 
            torch.cuda.empty_cache()
            
            for t in tqdm(range(self.inference_steps), total=self.inference_steps):

                def stable_diffusion_unet(latents, t, text_embed=None):
                    with torch.cuda.amp.autocast(enabled=True):
                        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        # if text_embed is not None:
                        #     prompt_embeds = text_embed 
                        
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds )[0]

                        # perform guidance
                        if self.do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                        
                        return noise_pred

                
                
                # print(">> self.eta: ", self.eta, self.alpha_prod_t_prevs, self.alpha_prod_ts)
                def decoding(x):
                    with torch.cuda.amp.autocast(enabled=True):
                        self.vae.grad=None
                        x.grad = None
                        # torch.cuda.empty_cache()
                        # import gc
                        # gc.collect()
                        print(">>>>> cuda vae memory111: ", torch.cuda.mem_get_info(device="cuda:0"), torch.cuda.memory_allocated(0))
                        temp = (self.vae.decode(x.permute(0, 2, 1, 3, 4).reshape(-1, x.shape[1], *x.shape[3:]) / self.vae.config.scaling_factor, return_dict=False, generator=self.generator)[0]).unsqueeze(0)
                        print(">>>>> cuda vae memory222: ", torch.cuda.mem_get_info(device="cuda:0"), torch.cuda.memory_allocated(0))
                        print(">>> temp: ", temp.shape) ## temp:  torch.Size([1, 15, 3, 256, 256])
                        return temp.squeeze()
                
                kwargs = { 'looplength': 4,  'text_embed': prompt_embeds} #, 'post_process': lambda x: decoding(x)}

                with torch.cuda.amp.autocast(enabled=True):
                    self.unet.grad=None
                    guidance.guider.vae.grad=None
                    self.vae.grad=None
                    guidance.guider.guider.model.grad = None
                    torch.cuda.empty_cache()
                    gc.collect()

                    outputs = guidance.guide_step(
                        latents, t, stable_diffusion_unet,
                        self.ts,
                        self.alpha_prod_ts, 
                        self.alpha_prod_t_prevs,
                        self.eta,
                        **kwargs
                    )

                    latents = outputs#["latent"]
                    # prompt_embeds = outputs["text_embeds"]

                    # latents = guidance.guide_step(
                    #     latents, t, stable_diffusion_unet,
                    #     self.ts,
                    #     self.alpha_prod_ts, 
                    #     self.alpha_prod_t_prevs,
                    #     self.eta,
                    #     **kwargs
                    # )
 
            image = self.decode_latents(latents) #self.decode(latents)

            # video_tensor = self.decode_latents(latents)
            # video = self.sd_pipeline.video_processor.postprocess_video(video=video_tensor, output_type="pil")
            # # frames = output.frames[0]
            # export_to_gif(video[0], "logs/ims_fwdvideo/animation_test2.gif")

            tot_samples.append(image.clone().cpu())
            tot_samples = torch.concat(tot_samples)
            videos = self.sd_pipeline.video_processor.postprocess_video(video=tot_samples, output_type="pil")
            # export_to_gif(videos[0], f"logs/ims_fwdvideo/animation_.gif")
            # export_to_video(videos[0], f"/mnt/data2/chris/outputs/vgg/test_seeinghearing/{self.text_id}.mp4", fps=8)
            print("asdasdasd >>>> ", f"{self.save_path}/{self.text_id}.mp4")
            export_to_video(videos[0], f"{self.save_path}/{self.text_id}.mp4", fps=8)
            # export_to_video(videos[0], f"/mnt/data2/chris/outputs/video_gen_guide/imagebind/mpgd/{self.text_id}.mp4", fps=8)
        
        return torch.concat(tot_samples)
        
    def tensor_to_obj(self, x):

        # print(">>> xx:: ", x.shape)
        
        videos = self.sd_pipeline.video_processor.postprocess_video(video=x, output_type="pil")
        # export_to_gif(videos[0], f"logs/ims_fwdvideo/animation_.gif")
        # export_to_video(videos[0], f"/mnt/data2/chris/outputs/vgg/test_seeinghearing/{self.text_id}.mp4", fps=8)
        export_to_video(videos[0], f"/mnt/data2/chris/outputs/vgg/test_seeingandhearing_textupd/{self.text_id}.mp4", fps=8)
     
        
        return videos
    
    def obj_to_tensor(self, objs: List[Image.Image]) -> torch.Tensor:
        '''
            convert a list of PIL images into tensors
        '''
        images = [to_tensor(pil_image) for pil_image in objs]
        tensor_images = torch.stack(images).to(self.device)
        return tensor_images * 2 - 1

# import math
# import torch
# from torchvision.transforms.functional import to_tensor
# from PIL import Image
# from tqdm import tqdm
# from typing import List

# from diffusers.utils.torch_utils import randn_tensor

# from utils.configs import Arguments
# from .base import BaseSampler
# from methods.base import BaseGuidance
# # from diffusers import StableDiffusionPipeline
# from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter, UNet2DConditionModel, UNetMotionModel,AutoencoderKL
# from  diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
# from utils.env_utils import *
# from diffusers.utils import export_to_gif, export_to_video
# from torch.cuda.amp import autocast

# class StableDiffusionVideoNormalSampler(BaseSampler):

#     def __init__(self, args: Arguments):

#         super(StableDiffusionVideoNormalSampler, self).__init__(args)
#         # self.video_size = args.video_size
#         self.inference_steps = args.inference_steps
#         self.eta = args.eta
#         self.log_traj = args.log_traj
#         self.generator = torch.manual_seed(self.seed)

#         # FIXME: need to send batch_id to guider
#         self.args = args
#         # prepare unet, prev_t, alpha_prod, alpha_prod_prev...
#         self._build_diffusion(args)

#     @torch.no_grad()
#     def decode(self, latents):
#         return self.sd_pipeline.vae.decode(latents / self.sd_pipeline.vae.config.scaling_factor, return_dict=False, generator=self.generator)[0]

#     @torch.no_grad()
#     def decode_latents(self, latents, decode_chunk_size: int = 16):
#         # latents = 1 / self.sd_pipeline.vae.config.scaling_factor * latents
#         with autocast(True):
#             # print(">>> latenttss: ", latents.shape) ### torch.Size([1, 4, 15, 32, 32])
#             batch_size, channels, num_frames, height, width = latents.shape
#             latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

#             video = []
#             # for i in range(0, latents.shape[0], decode_chunk_size):
#             #     batch_latents = latents[i : i + decode_chunk_size]
#             #     batch_latents = self.vae.decode(batch_latents).sample
#             #     video.append(batch_latents)
#             video = self.sd_pipeline.vae.decode(latents / self.sd_pipeline.vae.config.scaling_factor, return_dict=False, generator=self.generator)[0]
#             # video = torch.cat(video)
#             video = video[None, :].reshape((batch_size, num_frames, -1) + video.shape[2:]).permute(0, 2, 1, 3, 4)
#             # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
#             video = video.float()
#             return video

#     @torch.no_grad()
#     def _build_diffusion(self, args):
        
#         '''
#             Different diffusion models should be registered here
#         '''
#         # self.sd_pipeline = StableDiffusionPipeline.from_pretrained(args.model_name_or_path).to(self.device)
#         # self.sd_pipeline = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16) #StableDiffusionPipeline.from_pretrained(args.model_name_or_path).to(self.device)
#         adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
#         self.sd_pipeline = AnimateDiffPipeline.from_pretrained(args.model_name_or_path, motion_adapter=adapter, torch_dtype=torch.float16).to(self.device)
        
#         self.sd_pipeline.scheduler.set_timesteps(args.inference_steps)
#         self.scheduler = self.sd_pipeline.scheduler

#         self.video_processor = self.sd_pipeline.video_processor

#         unet_cond = UNet2DConditionModel.from_pretrained(args.model_name_or_path, subfolder="unet", torch_dtype=torch.float16).to(self.device)
#         unet = UNetMotionModel.from_unet2d(unet_cond, adapter).to(self.device)
        
#         # unet = self.sd_pipeline.unet
#         unet.eval()

#         for param in unet.parameters():
#             param.requires_grad = False

        
#         # self.scheduler.set_timesteps(args.inference_steps)
#         ts = self.scheduler.timesteps
#         print(">>> ts: ", len(ts))

#         alpha_prod_ts = self.scheduler.alphas_cumprod[ts]
#         alpha_prod_t_prevs = torch.cat([alpha_prod_ts[1:], torch.ones(1) * self.scheduler.final_alpha_cumprod])

#         print(">> self.sd_pipeline.unet.config.sample_size: ", self.sd_pipeline.unet.config.sample_size)
#         self.height = self.width = self.sd_pipeline.unet.config.sample_size * self.sd_pipeline.vae_scale_factor

#         self.height = self.height //2 #//4 *3
#         self.width = self.width //2 #//4 *3
#         self.num_frames = 12

#         # prepare prompts: str or List[str]
#         self.prompts = self._prepare_prompts(args)  

#         # FIXME: classifier-free guidance params
#         self.do_classifier_free_guidance = True
#         self.guidance_scale = 7.5

#         # check inputs. Raise error if not correct
#         self.sd_pipeline.check_inputs(self.prompts, self.height, self.width, callback_steps=None)

#         self.unet, self.ts, self.alpha_prod_ts, self.alpha_prod_t_prevs = unet, ts, alpha_prod_ts, alpha_prod_t_prevs

    
#     def _prepare_prompts(self, args):
        
#         if args.dataset == 'parti_prompts':
#             prompts = [line.strip() for line in open(PARTIPROMPOTS_PATH, 'r').readlines()][:args.num_samples]
#         else:
#             prompts = ["flower"] * args.num_samples
        
#         return prompts

#     @torch.no_grad()
#     def sample(self, sample_size: int, guidance: BaseGuidance):
        
#         tot_samples = []
#         n_batchs = math.ceil(sample_size / self.per_sample_batch_size)

#         # output = self.sd_pipeline(
#         #         prompt=(
#         #             "pickup truck2"
#         #         ),
#         #         negative_prompt="bad quality, worse quality",
#         #         num_frames=16,
#         #         guidance_scale=7.5,
#         #         num_inference_steps=25,
#         #         generator=torch.Generator("cpu").manual_seed(42),
#         #     )
#         # frames = output.frames[0]
#         # print(">>>>> shape: ", len(frames))
#         # export_to_gif(frames, "logs/ims_fwdvideo/animation_test.gif")

#         for batch_id in range(n_batchs):
            
#             self.args.batch_id = batch_id

#             prompts = self.prompts[batch_id * self.per_sample_batch_size: min((batch_id + 1) * self.per_sample_batch_size, len(self.prompts))]

#             # encode input prompts
#             # print(">> prompts: ", prompts, self.device)
#             prompt_embeds, negative_prompt_embeds = self.sd_pipeline.encode_prompt(
#                 prompts, 
#                 self.device,
#                 num_images_per_prompt=1,
#                 do_classifier_free_guidance=self.do_classifier_free_guidance,
#                 negative_prompt=["bad quality, worse quality"]
#             )

#             if self.do_classifier_free_guidance:
#                 prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            
#             prompt_embeds = prompt_embeds.repeat_interleave(repeats=self.num_frames, dim=0)

#             latents = self.sd_pipeline.prepare_latents(
#                 len(prompts),
#                 self.sd_pipeline.unet.config.in_channels,
#                 self.num_frames,
#                 self.height,
#                 self.width,
#                 prompt_embeds.dtype,
#                 self.device,
#                 generator=self.generator
#             )

#             # self.sd_pipeline = None

#             for t in tqdm(range(self.inference_steps), total=self.inference_steps):
                
#                 with autocast(True):
#                     def stable_diffusion_unet(latents, t):
                        
#                             latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
#                             latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

#                             # print(">>> : ", latent_model_input.shape, prompt_embeds.shape) ## torch.Size([2, 4, 15, 32, 32])

#                             #noise_pred = unet(
#                             #     latent_input[:, :].float(),
#                             #     t.float(),
#                             #     encoder_hidden_states=prompt_embeds.float(),
#                             #     cross_attention_kwargs=None,
#                             #     added_cond_kwargs=None,
#                             # ).sample

#                             noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds)[0]

#                             # perform guidance
#                             if self.do_classifier_free_guidance:
#                                 noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#                                 noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                            
#                             return noise_pred

#                     # noise_pred = stable_diffusion_unet(latents,t)
#                     print("NORMALLL:::>>>> ")
#                     kwargs = {"text":prompts}
#                     latents = guidance.guide_step(
#                         latents, t, stable_diffusion_unet,
#                         self.ts,
#                         self.alpha_prod_ts, 
#                         self.alpha_prod_t_prevs,
#                         self.eta,
#                         **kwargs
#                     )

#                 # latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
#                 # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

#                 # tt = self.scheduler.timesteps[t]
#                 # print(">>> : ", latent_model_input.shape, prompt_embeds.shape) ## torch.Size([2, 4, 15, 32, 32])

#                 # noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=prompt_embeds).sample

#                 # # perform guidance
#                 # # if self.do_classifier_free_guidance:
#                 # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#                 # noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
                
#                 # print(">>> noise_pred: ", noise_pred.shape, latents.shape)
#                 # latents = self.scheduler.step(noise_pred, tt, latents).prev_sample

#             # latents = 1/(self.scheduler.alphas_cumprod[t].to(self.device) ** 0.5) * (latents - (1-self.scheduler.alphas_cumprod[t].to(self.device))**0.5 * noise_pred) 

#             image = self.decode_latents(latents) #self.decode(latents)

#             # video_tensor = self.decode_latents(latents)
#             # video = self.sd_pipeline.video_processor.postprocess_video(video=video_tensor, output_type="pil")
#             # # frames = output.frames[0]
#             # export_to_gif(video[0], "logs/ims_fwdvideo/animation_test2.gif")

#             tot_samples.append(image.clone().cpu())
        
#         return torch.concat(tot_samples)
        
#     def tensor_to_obj(self, x):

#         # print(">>> xx:: ", x.shape)
        
#         videos = self.sd_pipeline.video_processor.postprocess_video(video=x, output_type="pil")
#         # export_to_gif(videos[0], f"logs/ims_fwdvideo/animation_.gif")
#         export_to_video(videos[0], f"logs/ims_fwdvideo/animation_dogbubble_.mp4", fps=4)
#         # images = (x / 2 + 0.5).clamp(0, 1)
#         # images = images.cpu().permute(0, 2, 3, 1).numpy()
        
#         # if images.ndim == 3:
#         #     images = images[None, ...]
#         # images = (images * 255).round().astype("uint8")
#         # if images.shape[-1] == 1:
#         #     pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
#         # else:
#         #     pil_images = [Image.fromarray(image) for image in images]
        
#         return videos
    
#     def obj_to_tensor(self, objs: List[Image.Image]) -> torch.Tensor:
#         '''
#             convert a list of PIL images into tensors
#         '''
#         images = [to_tensor(pil_image) for pil_image in objs]
#         tensor_images = torch.stack(images).to(self.device)
#         return tensor_images * 2 - 1