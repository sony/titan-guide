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

class StableDiffusionVideoSampler(BaseSampler):

    def __init__(self, args: Arguments):

        super(StableDiffusionVideoSampler, self).__init__(args)
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
        # self.sd_pipeline = StableDiffusionPipeline.from_pretrained(args.model_name_or_path).to(self.device)
        # self.sd_pipeline = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16) #StableDiffusionPipeline.from_pretrained(args.model_name_or_path).to(self.device)
        adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float32)
        unet_cond = UNet2DConditionModel.from_pretrained(args.model_name_or_path, subfolder="unet", torch_dtype=torch.float32, local_files_only=True).to(self.device)
        # unet_cond = UNet2DConditionModel.from_pretrained("/mnt/data2/chris/hg_models/emilianJR--epiCRealism/unet/", torch_dtype=torch.float32, local_files_only=True).to(self.device)
        unet = UNetMotionModel.from_unet2d(unet_cond, adapter).to(dtype=torch.float32, device=self.device)
        
        self.sd_pipeline = AnimateDiffPipeline.from_pretrained(args.model_name_or_path, motion_adapter=adapter, torch_dtype=torch.float32).to(self.device)
        
        self.sd_pipeline.scheduler.set_timesteps(args.inference_steps)
        self.scheduler = self.sd_pipeline.scheduler

        self.video_processor = self.sd_pipeline.video_processor

        self.vae = AutoencoderKL.from_pretrained(args.model_name_or_path, subfolder="vae", torch_dtype=torch.float32).to(self.device) #sd = AnimateDiffPipeline.from_pretrained(self.args.model_name_or_path, motion_adapter=adapter, torch_dtype=torch.float16)
        # self.vae = vae#.to(self.args.device)
        self.vae.eval()

        # unet_cond = UNet2DConditionModel.from_pretrained(args.model_name_or_path, subfolder="unet", torch_dtype=torch.float32).to(self.device)
        # unet = UNetMotionModel.from_unet2d(unet_cond, adapter).to(self.device)

        
        
        # unet = self.sd_pipeline.unet
        unet.eval()

        for param in unet.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False

        
        # self.scheduler.set_timesteps(args.inference_steps)
        ts = self.scheduler.timesteps
        print(">>> ts: ", len(ts))

        alpha_prod_ts = self.scheduler.alphas_cumprod[ts]
        alpha_prod_t_prevs = torch.cat([alpha_prod_ts[1:], torch.ones(1) * self.scheduler.final_alpha_cumprod])

        print(">> self.sd_pipeline.unet.config.sample_size: ", self.sd_pipeline.unet.config.sample_size)
        self.height = self.width = self.sd_pipeline.unet.config.sample_size * self.sd_pipeline.vae_scale_factor

        self.height = 256 #384 #320 #self.height//4 *3 # //2 #//4 *3 #//2 #//4 *3
        self.width = 256 #384 #320 #self.width//4 *3#//2 #//4 *3 #//2 #//4 *3
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


            if os.path.exists(f"{self.save_path}/{self.text_id}.mp4"):
                continue

            # encode input prompts
            # print(">> prompts: ", prompts, self.device)
            self.sd_pipeline.to(self.device)
            prompt_embeds, negative_prompt_embeds = self.sd_pipeline.encode_prompt(
                prompts, #[0], ##
                self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=["bad quality, worse quality, cartoon"]
            )
            self.sd_pipeline.to("cpu")

            torch.backends.cuda.enable_mem_efficient_sdp(False) ### SAKTI
            # self.

            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds.detach(), prompt_embeds.detach()])
            
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
          

            # self.sd_pipeline = None
            if hasattr(guidance.guider, 'prepare_target'): ### means it is imagebind guidance
                AUDIOPATH='./dataset/vggsound/svg_no/audio2video/audio'
                #AUDIOPATH='/groups/gce50978/acg16791sz/data/vggsound/svg_no/audio2video/audio'
                target = {'audio': f'{AUDIOPATH}/{self.text_id}.wav', 'text': prompts}
                # print("???>>> guidance: ", guidance.guider)
                guidance.guider.prepare_target(target, device=self.device)
            
            tot_samples = [] 
 
            
            for t in tqdm(range(self.inference_steps), total=self.inference_steps):

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
 
                kwargs = { 'looplength': 2}
                
                # print(">> self.eta: ", self.eta, self.alpha_prod_t_prevs, self.alpha_prod_ts)
                latents = guidance.guide_step(
                    latents, t, stable_diffusion_unet, self.vae,
                    self.ts,
                    self.alpha_prod_ts, 
                    self.alpha_prod_t_prevs,
                    self.eta,
                    **kwargs
                )

                torch.cuda.empty_cache()
                
            image = self.decode_latents(latents) #self.decode(latents)
 
            tot_samples.append(image.clone().cpu())
            tot_samples = torch.concat(tot_samples)
            
            videos = self.sd_pipeline.video_processor.postprocess_video(video=tot_samples, output_type="pil")
           
            print(">>>save at: ", f"{self.save_path}/{self.text_id}.mp4")
            export_to_video(videos[0], f"{self.save_path}/{self.text_id}.mp4", fps=8)
 

        
        return torch.concat(tot_samples)
        
    def tensor_to_obj(self, x):

        # print(">>> xx:: ", x.shape)
        
        videos = self.sd_pipeline.video_processor.postprocess_video(video=x, output_type="pil")
  
        export_to_video(videos[0], f"./outputs/vgg/test_seeingandhearing_textupd/{self.text_id}.mp4", fps=8)
        
        return videos
    
    def obj_to_tensor(self, objs: List[Image.Image]) -> torch.Tensor:
        '''
            convert a list of PIL images into tensors
        '''
        images = [to_tensor(pil_image) for pil_image in objs]
        tensor_images = torch.stack(images).to(self.device)
        return tensor_images * 2 - 1

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds