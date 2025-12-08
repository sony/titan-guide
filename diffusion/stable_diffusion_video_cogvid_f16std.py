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

from .cogvid_pipeline.pipeline_cogvideox import CogVideoX_Fun_Pipeline
from .cogvid_pipeline.transformer3d import CogVideoXTransformer3DModel
from .cogvid_pipeline.autoencoder_magvit import AutoencoderKLCogVideoX
from .cogvid_pipeline.pipeline_cogvideox_inpaint import CogVideoX_Fun_Pipeline_Inpaint
from transformers import T5EncoderModel, T5Tokenizer

class StableDiffusionVideoCogVidSampler(BaseSampler):

    def __init__(self, args: Arguments):

        super(StableDiffusionVideoCogVidSampler, self).__init__(args)
        # self.video_size = args.video_size
        self.inference_steps = args.inference_steps
        self.eta = args.eta
        self.log_traj = args.log_traj
        self.generator = torch.manual_seed(self.seed)

        # FIXME: need to send batch_id to guider
        self.args = args
        # prepare unet, prev_t, alpha_prod, alpha_prod_prev...
        self.save_path = args.save_path
        self.audio_path='/mnt/data2/dataset/vggsound/svg_no/audio2video/audio'
        self._build_diffusion(args)

        self.prompts = self._prepare_prompts(args)  

        self.weight_type = torch.float16

    @torch.no_grad()
    def decode(self, latents):
        return self.sd_pipeline.vae.decode(latents / self.sd_pipeline.vae.config.scaling_factor, return_dict=False, generator=self.generator)[0]

    @torch.no_grad()
    def decode_latents(self, latents, decode_chunk_size: int = 16):
        # latents = 1 / self.sd_pipeline.vae.config.scaling_factor * latents
        with autocast(True):
            latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
            latents = 1 / self.sd_pipeline.vae.config.scaling_factor * latents

            frames = self.vae.decode(latents).sample
            # frames = (frames / 2 + 0.5).clamp(0, 1)
            # frames = (frames / 2 + 0.5).clamp(0, 1)
            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
            # frames = frames#.cpu().float().numpy()
            # # print(">>> latenttss: ", latents.shape) ### torch.Size([1, 4, 15, 32, 32])
            # batch_size, channels, num_frames, height, width = latents.shape
            # latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

            # video = []
            # # for i in range(0, latents.shape[0], decode_chunk_size):
            # #     batch_latents = latents[i : i + decode_chunk_size]
            # #     batch_latents = self.vae.decode(batch_latents).sample
            # #     video.append(batch_latents)
            # video = self.sd_pipeline.vae.decode(latents / self.sd_pipeline.vae.config.scaling_factor, return_dict=False )[0]
            # # video = torch.cat(video)
            # video = video[None, :].reshape((batch_size, num_frames, -1) + video.shape[2:]).permute(0, 2, 1, 3, 4)
            # # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
            # video = video.float()
            return frames

    @torch.no_grad()
    def _build_diffusion(self, args):
        
        '''
            Different diffusion models should be registered here
        '''
        self.weight_type = torch.float16
        model_name = "/mnt/data2/chris/code/SDS-Bridge/2D_experiments/models/Diffusion_Transformer/CogVideoX-Fun-V1.1-2b-InP"
        vae = AutoencoderKLCogVideoX.from_pretrained(
            model_name, 
            subfolder="vae"
        ).to(self.weight_type).to(self.device)

        text_encoder = T5EncoderModel.from_pretrained(
            model_name, subfolder="text_encoder", torch_dtype=self.weight_type
        ).to(self.device)
        sampler_name = "DDIM_Origin"

        scheduler = DDIMScheduler.from_pretrained(
            model_name, 
            subfolder="scheduler"
        )

        transformer = CogVideoXTransformer3DModel.from_pretrained_2d(
            model_name, 
            subfolder="transformer",
        ).to(self.weight_type).to(self.device)

        tokenizer = T5Tokenizer.from_pretrained(model_name, subfolder="tokenizer")
        
        # self.sd_pipeline = AnimateDiffPipeline.from_pretrained(args.model_name_or_path, motion_adapter=adapter, torch_dtype=torch.float16).to(self.device)
        

        self.sd_pipeline = CogVideoX_Fun_Pipeline_Inpaint.from_pretrained(
            model_name,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=torch.float16
            # ,
            # torch_dtype=torch.bfloat16
        ).to(self.device)
        
        self.sd_pipeline.scheduler.set_timesteps(args.inference_steps)
        self.scheduler = self.sd_pipeline.scheduler

        print("??? .>>>> self.scheduler: ", self.scheduler)

        self.video_processor = self.sd_pipeline.video_processor


        unet = transformer.to(self.device) #self.sd_pipeline.transformer
        unet.eval()

        self.vae = vae

        for param in unet.parameters():
            param.requires_grad = False

        for param in self.vae.parameters():
            param.requires_grad = False

        
        # self.scheduler.set_timesteps(args.inference_steps)
        ts = self.scheduler.timesteps
        print(">>> ts: ", len(ts))

        alpha_prod_ts = self.scheduler.alphas_cumprod[ts]
        alpha_prod_t_prevs = torch.cat([alpha_prod_ts[1:], torch.ones(1) * self.scheduler.final_alpha_cumprod])
        

        print(">> self.sd_pipeline.unet.config.sample_size: ", self.sd_pipeline.transformer.config)
        self.height = self.width = 512#self.sd_pipeline.transformer.config.sample_size * self.sd_pipeline.vae_scale_factor

        self.height = 256 #self.height //4 *3 #//2 #//4 *3
        self.width = 384 #self.width //4 *3 #//2 #//4 *3
        self.num_frames = 32 #42#16 #32 #16

        # prepare prompts: str or List[str]
        self.prompts = self._prepare_prompts(args)  

        # FIXME: classifier-free guidance params
        self.do_classifier_free_guidance = True
        self.guidance_scale = 7.5 #7.5

        # check inputs. Raise error if not correct
        # self.sd_pipeline.check_inputs(self.prompts, self.height, self.width )

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
        # if args.dataset == 'parti_prompts':
        #     prompts = [line.strip() for line in open(PARTIPROMPOTS_PATH, 'r').readlines()][:args.num_samples]
        # else:
        #     prompts = ["flower"] * args.num_samples
        
        # return prompts

    @torch.no_grad()
    def sample(self, sample_size: int, guidance: BaseGuidance):
        
        tot_samples = []
        n_batchs = math.ceil(sample_size / self.per_sample_batch_size)

        # output = self.sd_pipeline(
        #         prompt=(
        #             "pickup truck2"
        #         ),
        #         negative_prompt="bad quality, worse quality",
        #         num_frames=16,
        #         guidance_scale=7.5,
        #         num_inference_steps=25,
        #         generator=torch.Generator("cpu").manual_seed(42),
        #     )
        # frames = output.frames[0]
        # print(">>>>> shape: ", len(frames))
        # export_to_gif(frames, "logs/ims_fwdvideo/animation_test.gif")

        for batch_id in range(n_batchs):
            
            self.args.batch_id = batch_id

            prompts = self.prompts[batch_id * self.per_sample_batch_size: min((batch_id + 1) * self.per_sample_batch_size, len(self.prompts))]

            # prompts = self.prompts[batch_id * self.per_sample_batch_size: min((batch_id + 1) * self.per_sample_batch_size, len(self.prompts))]
            print(">>> prompts: ", prompts)
            if self.ids is not None:
                ids = self.ids[batch_id * self.per_sample_batch_size: min((batch_id + 1) * self.per_sample_batch_size, len(self.prompts))]
                self.text_id = ids[0][:-2]#.replace("_0.",".")
            else:
                self.text_id = prompts

            # encode input prompts
            # print(">> prompts: ", prompts, self.device)
            self.sd_pipeline.to(self.device)
            prompt_embeds, negative_prompt_embeds = self.sd_pipeline.encode_prompt(
                prompts, 
                device=self.device,
                num_videos_per_prompt=1,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=["bad quality, worse quality"]
            )

            self.sd_pipeline.to("cpu")
            self.unet.to(self.device)
            self.vae.to(self.device)
            # torch.backends.cuda.enable_mem_efficient_sdp(False) ### SAKTI

            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            
            
            target = {'audio': f'{self.audio_path}/{self.text_id}.wav', 'text': prompts}


            if self.args[""] = "dover_aesthetic"
            guidance.guider.prepare_target(target, device=self.device)
            # print("???>>> guidance: ", guidance.guider)
            # prompt_embeds = prompt_embeds.repeat_interleave(repeats=self.num_frames, dim=0)

            # latents = self.sd_pipeline.prepare_latents(
            #     len(prompts),
            #     self.sd_pipeline.transformer.config.in_channels,
            #     self.num_frames,
            #     self.height,
            #     self.width,
            #     prompt_embeds.dtype,
            #     self.device,
            #     generator=self.generator
            # )[0]
            latents = torch.randn( [1, 4, 16, 32, 48], #[1, 8, 16, 32, 48],
                                generator=self.generator,
                                # device=,
                                dtype=self.weight_type,
                                requires_grad=False).to(self.device)

            print(">>> latents.,::: ", latents.shape, self.height, self.width ) ##    >> should be torch.randn( [1, 1, 16, 32, 48]
            
            # mask = torch.zeros_like(latents).to(latents.device, latents.dtype)
            # masked_video_latents = torch.zeros_like(latents).to(latents.device, latents.dtype)

            # mask_input = torch.cat([mask] * 2) if self.do_classifier_free_guidance else mask
            # masked_video_latents_input = (
            #     torch.cat([masked_video_latents] * 2) if self.do_classifier_free_guidance else masked_video_latents
            # )
            # inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=2).to(latents.dtype)

            mask_latents = torch.zeros_like(latents)[:, :, :1].to(latents.device, latents.dtype)
            masked_video_latents = torch.zeros_like(latents).to(latents.device, latents.dtype)

            mask_input = torch.cat([mask_latents] * 2) if self.do_classifier_free_guidance else mask_latents
            masked_video_latents_input = (
                torch.cat([masked_video_latents] * 2) if self.do_classifier_free_guidance else masked_video_latents
            )
            inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=2).to(latents.dtype)
            # self.sd_pipeline = None
            extra_step_kwargs = {}

            for t in tqdm(range(1, self.inference_steps), total=self.inference_steps):
                
                
                def stable_diffusion_unet(latents, t):
                    # with autocast(True):
                    with torch.no_grad():
                        # print(">> latents: ", latents.shape)
                        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        # print(">>> : ", latent_model_input.shape, prompt_embeds.shape) ## torch.Size([2, 4, 15, 32, 32])

                        timestep = t.expand(latent_model_input.shape[0]).to(latent_model_input.device)
                        # print(">>> inpaint_latents: ", inpaint_latents.shape,   prompt_embeds.shape,  ) ## torch.Size([2, 2, 16, 32, 48]) torch.Size([2, 226, 4096])
                        # print(">> latent_model_input: ", latent_model_input.shape, prompt_embeds.shape, timestep.shape, inpaint_latents.shape) ## torch.Size([2, 11, 16, 32, 32]) torch.Size([2, 226, 4096]) torch.Size([2]) torch.Size([2, 11, 17, 32, 32])
                        noise_pred = self.unet(#latent_model_input, t[0], 
                                    hidden_states=latent_model_input,#.float(),
                                    encoder_hidden_states=prompt_embeds,#.float(),
                                    timestep=timestep,#,.float(),
                                    image_rotary_emb=None,
                                    return_dict=False,
                                    inpaint_latents=inpaint_latents)[0]

                        # noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds)[0]

                        # perform guidance
                        if self.do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                        
                        return noise_pred

                # with autocast(True):
                #     # noise_pred = stable_diffusion_unet(latents,t)
                #     kwargs = {"text":prompts}

                #     latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                #     latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                #     tt = self.ts[t]
                #     timestep = tt.expand(latent_model_input.shape[0]).to(latent_model_input.device)
                #     # print(">>> inpaint_latents: ", inpaint_latents.shape,   prompt_embeds.shape,  ) ## torch.Size([2, 2, 16, 32, 48]) torch.Size([2, 226, 4096])
                #     print(">> latent_model_input: ", latent_model_input.shape, prompt_embeds.shape, timestep.shape, inpaint_latents.shape) ## torch.Size([2, 11, 16, 32, 32]) torch.Size([2, 226, 4096]) torch.Size([2]) torch.Size([2, 11, 17, 32, 32])
                #     noise_pred = self.unet(#latent_model_input, t[0], 
                #                 hidden_states=latent_model_input,
                #                 encoder_hidden_states=prompt_embeds.float(),
                #                 timestep=timestep.float(),
                #                 image_rotary_emb=None,
                #                 return_dict=False,
                #                 inpaint_latents=inpaint_latents)[0]

                #     # noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds)[0]

                #     # perform guidance
                #     if self.do_classifier_free_guidance:
                #         noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                #         noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                #     extra_step_kwargs["generator"] = self.generator
                #     latents = self.scheduler.step(noise_pred, tt, latents, **extra_step_kwargs, return_dict=False)[0]
                # latents = self.scheduler.step(noise_pred, tt, latents, **extra_step_kwargs, return_dict=False)[0]
                kwargs = {"text":prompts, "cogvid": 1}
                # print(">> self.eta: ", self.eta, self.alpha_prod_t_prevs, self.alpha_prod_ts)
                    #         x: torch.Tensor,
                    # t: int,
                    # unet: torch.nn.Module,
                    # vae: torch.nn.Module,
                    # ts: torch.LongTensor,
                    # alpha_prod_ts: torch.Tensor,
                    # alpha_prod_t_prevs: torch.Tensor,
                    # eta: float,

                # latents = stable_diffusion_unet(latents, self.ts[t])

                #### EDITED
                latents = guidance.guide_step(
                    latents, t, stable_diffusion_unet, self.vae,
                    self.ts,
                    self.alpha_prod_ts, 
                    self.alpha_prod_t_prevs,
                    self.eta,
                    **kwargs
                )

                # latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # tt = self.scheduler.timesteps[t]
                # print(">>> : ", latent_model_input.shape, prompt_embeds.shape) ## torch.Size([2, 4, 15, 32, 32])

                # noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=prompt_embeds).sample

                # # perform guidance
                # # if self.do_classifier_free_guidance:
                # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                # noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
                
                # print(">>> noise_pred: ", noise_pred.shape, latents.shape)
                # latents = self.scheduler.step(noise_pred, tt, latents).prev_sample

            # latents = 1/(self.scheduler.alphas_cumprod[t].to(self.device) ** 0.5) * (latents - (1-self.scheduler.alphas_cumprod[t].to(self.device))**0.5 * noise_pred) 

            image = self.decode_latents(latents)  #self.decode(latents)

            tot_samples = [] 
            tot_samples.append(image.clone().cpu())
            tot_samples = torch.concat(tot_samples)
            videos = self.sd_pipeline.video_processor.postprocess_video(video=tot_samples, output_type="pil")
            # export_to_gif(videos[0], f"logs/ims_fwdvideo/animation_.gif")
            # export_to_video(videos[0], f"/mnt/data2/chris/outputs/vgg/test_seeinghearing/{self.text_id}.mp4", fps=8)
            print("SAVE path >>>> ", f"{self.save_path}/{self.text_id}.mp4")
            export_to_video(videos[0], f"{self.save_path}/{self.text_id}.mp4", fps=8)

            # print(">> image: ", image.shape) ##  torch.Size([1, 3, 16, 256, 384])
            # self.tensor_to_obj(image)
            # video_tensor = self.decode_latents(latents)
            # video = self.sd_pipeline.video_processor.postprocess_video(video=video_tensor, output_type="pil")
            # # frames = output.frames[0]
            # export_to_gif(video[0], "logs/ims_fwdvideo/animation_test2.gif")

            # tot_samples.append(image.clone().cpu().float())
        
        return torch.concat(tot_samples)#.numpy()
        
    def tensor_to_obj(self, x):

        print(">>> xx:: ", x.shape)
        
        videos = self.sd_pipeline.video_processor.postprocess_video(video=x, output_type="pil")
        # export_to_gif(videos[0], f"logs/ims_fwdvideo/animation_.gif")
        export_to_video(videos[0], f"logs/ims_fwdvideo/animation_dograin_temporal_.mp4", fps=7)
        # images = (x / 2 + 0.5).clamp(0, 1)
        # images = images.cpu().permute(0, 2, 3, 1).numpy()
        
        # if images.ndim == 3:
        #     images = images[None, ...]
        # images = (images * 255).round().astype("uint8")
        # if images.shape[-1] == 1:
        #     pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        # else:
        #     pil_images = [Image.fromarray(image) for image in images]
        
        return videos
    
    def obj_to_tensor(self, objs: List[Image.Image]) -> torch.Tensor:
        '''
            convert a list of PIL images into tensors
        '''
        images = [to_tensor(pil_image) for pil_image in objs]
        tensor_images = torch.stack(images).to(self.device)
        return tensor_images * 2 - 1