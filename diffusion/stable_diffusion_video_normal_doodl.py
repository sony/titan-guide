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
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from utils.env_utils import *
from diffusers.utils import export_to_gif, export_to_video
from torch.cuda.amp import autocast
import numpy as np
import random


from .doodl.helper_functions import *
from .doodl.memcnn import InvertibleModuleWrapper


class StableDiffusionVideoDoodlSampler(BaseSampler):

    def __init__(self, args: Arguments):

        super(StableDiffusionVideoDoodlSampler, self).__init__(args)
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
        return self.vae.decode(latents / self.sd_pipeline.vae.config.scaling_factor, return_dict=False, generator=self.generator)[0]

    # @torch.no_grad()
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
            # video = self.vae.decode(latents.half() / self.vae.config.scaling_factor, return_dict=False, generator=self.generator)[0]
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
        self.sd_pipeline = AnimateDiffPipeline.from_pretrained(args.model_name_or_path, motion_adapter=adapter, torch_dtype=self.data_type).to(self.device)
        
        self.sd_pipeline.scheduler.set_timesteps(args.inference_steps)
        self.scheduler = self.sd_pipeline.scheduler
        self.scheduler.set_timesteps(args.inference_steps)

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
        
        
        ts = self.scheduler.timesteps
        print(">>> ts: ", len(ts))

        alpha_prod_ts = self.scheduler.alphas_cumprod[ts]
        alpha_prod_t_prevs = torch.cat([alpha_prod_ts[1:], torch.ones(1) * self.scheduler.final_alpha_cumprod])

        print(">> self.sd_pipeline.unet.config.sample_size: ", self.sd_pipeline.unet.config.sample_size)
        self.height = self.width = self.sd_pipeline.unet.config.sample_size * self.sd_pipeline.vae_scale_factor

        self.height = self.height //2 #//4 *3 #//2 #//4 *3
        self.width = self.width //2 #//4 *3 #//2 #//4 *3
        self.num_frames = 11 #16 #14 #16 #32 #16

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

    # @torch.no_grad()
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

                # print(">>> self.text_prompt: ", self.text_id)

                self.text_id = self.text_id[0].replace(".","").replace(",","").replace("|","").replace("\"","").replace("'","")

            text_id = self.text_id
            if os.path.exists(f"{self.save_path}/{self.text_id}.mp4"):
                continue
            # prompts = prompts[0][1]

            # if os.path.exists(f"/mnt/data2/chris/outputs/vgg/test_seeingandhearing_textupd/{self.text_id}.mp4"):
            #     continue

            print(">>> self.text_promptsss: ", text_id)
            # encode input prompts
            # print(">> prompts: ", prompts, self.device)
            # with torch.no_grad():
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
            
            # del self.sd_pipeline
            
            # self.sd_pipeline = None
            if 'imagebind' in self.args.task:
                AUDIOPATH='/mnt/data2/dataset/vggsound/svg_no/audio2video/audio'
                target = {'audio': f'{AUDIOPATH}/{self.text_id}.wav', 'text': prompts}
                # print("???>>> guidance: ", guidance.guider)
                guidance.guider.prepare_target(target, device=self.device)
            elif 'video_inpainting' in self.args.task :#hasattr(guidance.guider, 'prepare_target_video'): ### means it is imagebind guidance
                VIDEOPATH=self.args.target#'/mnt/data2/dataset/vggsound/svg_no/audio2video/audio'
                target = f'{VIDEOPATH}/{self.text_id}.mp4'#{'video': f'{AUDIOPATH}/{self.text_id}.mp4', 'text': prompts}
                # print("???>>> guidance: ", guidance.guider)
                print(">>>>> TARGETss : ", target)
                with torch.no_grad():
                    frame_target = guidance.guider.prepare_target(target, device=self.device)
                    encode_vid = self.vae.encode(frame_target.half(), return_dict=False)[0].sample() * self.vae.config.scaling_factor 
                    # encode_vid = self.vae.encode(frame_target.half(), return_dict=False)[0].sample() * self.vae.config.scaling_factor 
                    encode_vid = encode_vid.transpose(0, 1).unsqueeze(0) ###  torch.Size([16, 4, 32, 32])
                    start_idx = 0
                    end_idx = latents.shape[2]
                    encode_vid = encode_vid[:, : , :end_idx]
                    print(">> start end idxs: ", start_idx, end_idx, encode_vid.shape)

            self.sd_pipeline= self.sd_pipeline.to("cpu")
            tot_samples = [] 
            torch.cuda.empty_cache()

            mix_weight = 0.93
            keep_input = True
            tied_latents=True
            single_variable=False
            renormalize_latents=True

            mix = MixingLayer(mix_weight).to(self.device)
            self.mix = InvertibleModuleWrapper(mix, keep_input=keep_input, keep_input_inverse=keep_input, num_bwd_passes=1)


            keep_input = True
            tied_latents=True
            single_variable=False
            renormalize_latents=True
            sd_guidance_scale = 7.5 
            # steps = 
            embedding_unconditional, embedding_conditional = prompt_embeds.chunk(2)
            s = SteppingLayer(  self.unet,
                                embedding_unconditional, #.repeat_interleave(repeats=self.num_frames, dim=0).detach(),
                                embedding_conditional, #.repeat_interleave(repeats=self.num_frames, dim=0).detach(),
                                # inpainting_latents=inpaint_latents,
                                guidance_scale=sd_guidance_scale,
                                num_timesteps=self.inference_steps,
                                clip_cond_fn=None,
                                single_variable=single_variable,
                                scheduler=self.scheduler).to(self.device)
            s = InvertibleModuleWrapper(s, keep_input=keep_input,
                                                keep_input_inverse=keep_input,
                                num_bwd_passes=1)
            self.doodl_s = s
            self.timesteps = s._fn.scheduler.timesteps

            print(">>s._fn.scheduler.timesteps: ", len(s._fn.scheduler.timesteps))

            # print("latentslatents : ", latents.shape, encode_vid.shape)
            if 'video_inpainting' in self.args.task :
                latents[:start_idx+2] = encode_vid[:start_idx+2]
                latents[end_idx-2:] = encode_vid[end_idx-2:]
            input_latent_pair = torch.cat([latents.clone(), latents.clone()])
            


            ### TEST
            # tt = self.ts[1]
            # latent_model_input = latents #torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, tt)
            # pe =  prompt_embeds[:16]
            # epsilon = self.unet(latent_model_input, tt, pe) ## torch.Size([1, 4, 16, 32, 32]) torch.Size([32, 77, 768])
            # print(">>> PASSED: ", latent_model_input.shape, prompt_embeds.shape, tt, pe.shape) #  torch.Size([2, 4, 16, 32, 32]) torch.Size([32, 77, 768])
            # with torch.enable_grad():
            #     lasss = torch.rand(2,1).requires_grad_(True)
            #     lasss.mean().backward()

            

            with torch.enable_grad(): 
                for up_step in range(2):
                    with torch.cuda.amp.autocast(enabled=True):
                        input_latent_pair.grad = None
                        input_latent_pair = input_latent_pair.detach().clone().requires_grad_(True)
                        s.zero_grad()
                        # s.grad=None
                        # mix.grad = None
                        # self.vae.grad=None
                        # input_latent_pair.grad = None
                        # print(">> input_latent_pair: ", input_latent_pair.requires_grad)
                        # ccc =  input_latent_pair.mean()
                        # ccc.backward()
                        # print("loss11: ", input_latent_pair.grad)
                        input_latent_pair_out = input_latent_pair
                        for t in tqdm(range(self.inference_steps), total=self.inference_steps):

                
                            orig_norm = latents.norm().item()
                            # with torch.cuda.amp.autocast(enabled=True):
                                # self.unet.grad=None
                                # guidance.guider.vae.grad=None
                                # self.vae.grad=None
                                # guidance.guider.guider.model.grad = None
                                # torch.cuda.empty_cache()
                                # gc.collect()
                            # epsilon = unet(x, t, text_embed)
                            # x_prev = self._predict_x_prev_from_eps(x, epsilon, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs)
                            
                            # print("input_latent_pair: ", input_latent_pair.shape)
                            i = torch.tensor([t],
                                                dtype=torch_dtype,
                                                device=input_latent_pair.device)
                            tt = self.ts[t]
                            tt = torch.tensor([tt],
                                                dtype=torch_dtype,
                                                device=input_latent_pair.device)

                            # input_latent_pair_out = self.scheduler.scale_model_input(input_latent_pair_out, tt)
                            # bbb = input_latent_pair.chunk(2)
                            # loss = bbb[0].mean()
                            # loss.backward()
                            # print(">>>> input22: ", input_latent_pair.grad)
                            i, tt, input_latent_pair_out = s(i, tt, input_latent_pair_out)
                            

                            input_latent_pair_out = mix(input_latent_pair_out)

                            # bbb = input_latent_pair_out.chunk(2)
                            # loss = bbb[0].mean()
                            # loss.backward()
                            # print(">>>> input: ", input_latent_pair.grad)

                        print(">>> input_latent_pair_out: ", input_latent_pair_out.shape )
                        ims = [self.decode_latents(l.to(self.vae.dtype))#.sample
                                for l in input_latent_pair_out.chunk(2)]

                        # print(">> ims: ", ims[0].shape)
                        videos = ims[0]#.unsqueeze(0).permute(0, 2, 1, 3, 4).float().cpu().detach()
                        # videos = self.decode_latents(latents) #self.decode(latents)
                        # videos = self.sd_pipeline.video_processor.postprocess_video(video=videos.detach().cpu(), output_type="pil")
                        # print("asdasdasd >>>> ", f"{self.save_path}/{self.text_id}.mp4")
                        # export_to_video(videos[0], f"{self.save_path}/{self.text_id}.mp4", fps=8)
                        # loss = videos.mean()
                        # loss.backward()
                        # print("loss11: ", input_latent_pair.grad)

                        videos = videos.transpose(1, 2)
                        loss = guidance.guide_step(videos, return_logp=True, post_process=None)
                        loss.backward() ## backward in guide_step
                        print("LOSS : ", loss)
                        # print(">>> GRADDD: ", input_latent_pair.grad)
                        # print("loss22: ", input_latent_pair.grad)

                        input_latent_pair = self.update_gradient(input_latent_pair, orig_norm)
                        torch.cuda.empty_cache()

            # video_tensor = self.decode_latents(latents)
            # video = self.sd_pipeline.video_processor.postprocess_video(video=video_tensor, output_type="pil")
            # # frames = output.frames[0]
            # export_to_gif(video[0], "logs/ims_fwdvideo/animation_test2.gif")

            # tot_samples.append(ims[0].clone().cpu())
            # print(">>> encode_vid: ", ims[0].shape, encode_vid.shape) ##  torch.Size([1, 3, 12, 256, 256]) torch.Size([1, 4, 12, 32, 32])
            
            # ims2 = self.decode_latents( encode_vid.to(self.vae.dtype) )
            # print(">>>>ims2: ", ims2.shape) ## torch.Size([1, 3, 12, 256, 256])
            print(">>>>ims2  : ", ims[0].shape)
            tot_samples.append(ims[0].clone().cpu())
            tot_samples = torch.concat(tot_samples)
            videos = self.sd_pipeline.video_processor.postprocess_video(video=tot_samples, output_type="pil")
            # export_to_gif(videos[0], f"logs/ims_fwdvideo/animation_.gif")
            # export_to_video(videos[0], f"/mnt/data2/chris/outputs/vgg/test_seeinghearing/{self.text_id}.mp4", fps=8)
            print("savepath >>>> ", f"{self.save_path}/{text_id}.mp4")
            export_to_video(videos[0], f"{self.save_path}/{text_id}.mp4", fps=8)
            # export_to_video(videos[0], f"/mnt/data2/chris/outputs/video_gen_guide/imagebind/mpgd/{self.text_id}.mp4", fps=8)
        
        return torch.concat(tot_samples)
    
    def update_gradient(self, orig_latent_pair, orig_norm):
        tied_latents=True
        single_variable=False
        renormalize_latents=True

        grad = -0.5 * orig_latent_pair.grad

            # Average gradients if tied_latents
        if tied_latents:
            # print(">> grad: ", orig_latent_pair.grad)
            grad = grad.mean(dim=0, keepdim=True)
            grad = grad.repeat(2, 1, 1, 1, 1) ## torch.Size([2, 4, 2, 64, 64])

        new_latents = []
        lr = 5e-1 #0.01#0.2 #0.1#1 #8.2
        clip_grad_val = 1e-3
        use_momentum = False
        # doing perturbation linked as well
        # Perturbation is just random noise added
        perturbation = 0 #perturb_grad_scale * torch.randn_like(orig_latent_pair[0]) if perturb_grad_scale else 0
        
        # SGD step (S=stochastic from multicrop, can also just be GD)
        # Iterate through latents/grads
        for grad_idx, (g, l) in enumerate(zip(grad.chunk(2), orig_latent_pair.chunk(2))):
            
            # Clip max magnitude
            if clip_grad_val is not None:
                g = g.clip(-clip_grad_val, clip_grad_val)
                
            # SGD code
            # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
            # if use_momentum: 
            #     mom = 0.9
            #     # LR is grad scale
            #     # sticking with generic 0.9 momentum for now, no dampening
            #     if m==0:
            #         b = g
            #     else:
            #         b = mom * prev_b_arr[grad_idx] + g
            #     if use_nesterov:
            #         g = g + mom * b
            #     else:
            #         g = b
            #     prev_b_arr[grad_idx] = b.clone()
            new_l = l - (lr*g + perturbation)
            new_latents.append(new_l.clone())
        if tied_latents:  # don't think is needed with other tied_latent logic but just being safe
            combined_l = 0.5 * (new_latents[0] + new_latents[1])
            latent_pair = combined_l.repeat(2, 1, 1, 1, 1)
        else:
            latent_pair = torch.cat(new_latents)
            
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        # print("new latent pair>>> : ", latent_pair.shape)
        # if renormalize_latents: # Renormalize latents 
        for norm_i in range(2):
            latent_pair[norm_i] = latent_pair[norm_i] * orig_norm / latent_pair[norm_i].norm().item()

        return latent_pair
    
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

import torch.nn as nn

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

from torch import nn
# import torch
from diffusers import DDIMScheduler
from .doodl.helper_functions import *
# import memcnn

# from torch import nn

torch.autograd.set_detect_anomaly(False)
torch.set_grad_enabled(False)

class SteppingLayer(nn.Module):
    """
    This is a layer that performs DDIM stepping that will be wrapped
    by memcnn to be invertible
    """
    
    def __init__(self, unet,
                 embedding_uc,
                 embedding_c,
                 scheduler=None,
                 num_timesteps=50,
                 guidance_scale=7.5,
                 clip_cond_fn=None,
                 single_variable=False
                ):
        super(SteppingLayer, self).__init__()
        self.unet = unet
        self.e_uc = embedding_uc
        self.e_c = embedding_c
        self.num_timesteps = num_timesteps
        self.guidance_scale = guidance_scale
        if scheduler is None:
            self.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                                         beta_schedule="scaled_linear",
                                      num_train_timesteps=1000,
                                         clip_sample=False,
                                      set_alpha_to_one=False)
        else:
            self.scheduler = scheduler
        # self.scheduler.set_timesteps(num_timesteps)
        
        self.clip_cond_fn = clip_cond_fn
        
        self.single_variable = single_variable
        
    def zero_grad(self):
        for var in self.unet.parameters():
            var.grad = None

    def forward(self, i, t, latent_pair,
               reverse=False):
        """
        Run an EDICT step
        """
        for base_latent_i in range(2): 
            # Need to alternate order compatibly forward and backward
            if reverse:
                orig_i = self.num_timesteps - (i+1) 
                offset = (orig_i+1) % 2
                latent_i = (base_latent_i + offset) % 2
            else:
                offset = i%2
                latent_i = (base_latent_i + offset) % 2

            # leapfrog steps/run baseline logic hardcoded here
            latent_j = ((latent_i+1) % 2)
            
            latent_i = latent_i.long()
            latent_j = latent_j.long()
            
            if self.single_variable:
                # If it's the single variable baseline then just operate on one tensor
                latent_i = torch.zeros(1, dtype=torch.long).to(device)
                latent_j = torch.zeros(1, dtype=torch.long).to(device)

            # select latent model input
            if base_latent_i==0:
                latent_model_input = latent_pair.index_select(0, latent_j)
            else:
                latent_model_input = first_output
            latent_base = latent_pair.index_select(0, latent_i)

            cross_attention_kwargs = None
            added_cond_kwargs =None

            # print(">>> stepping layer: ", latent_model_input.shape, self.e_uc.shape , reverse) ## torch.Size([1, 4, 16, 64, 64])
            #Predict the unconditional noise residual
            # noise_pred_uncond = self.unet(latent_model_input, t[0], 
            #                          encoder_hidden_states=self.e_uc).sample
            # print(">> latent_model_input: ", latent_model_input.shape, t, self.e_uc.shape)
            noise_pred_uncond = self.unet(
                        latent_model_input,
                          t[0],
                        encoder_hidden_states=self.e_uc,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample                         

            # print(">> latent_model_input22: ", latent_model_input.shape, t, self.e_c.shape)
            #Predict the conditional noise residual and save the cross-attention layer activations
            noise_pred_cond = self.unet(
                        latent_model_input,
                         t[0],
                        encoder_hidden_states=self.e_c,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample
            # Get classifier free guidance term
            grad = (noise_pred_cond - noise_pred_uncond)
            noise_pred = noise_pred_uncond + self.guidance_scale * grad
            
            # incorporate classifier guidance if applicable
            if self.clip_cond_fn is not None:
                clip_grad = self.clip_cond_fn(latent_model_input, t.long(), 
                                             scheduler=self.scheduler)
                alpha_prod_t, beta_prod_t = get_alpha_and_beta(t.long(), self.scheduler)
                fac = beta_prod_t ** 0.5 
                noise_pred = noise_pred - fac * clip_grad 


            # Going forward or backward?
            step_call = reverse_step if reverse else forward_step
            # Step
            new_latent = step_call(self.scheduler,
                                      noise_pred,
                                        t[0].long(),
                                        latent_base)
            new_latent = new_latent.to(latent_base.dtype)

            
            # format outputs using index order
            if self.single_variable:
                combined_outputs = torch.cat([new_latent, new_latent])
                break
            
            if base_latent_i == 0: # first pass
                first_output = new_latent
            else: # second pass
                second_output = new_latent
                if latent_i==1: # so normal order
                    combined_outputs = torch.cat([first_output, second_output])
                else: # Offset so did in reverse
                    combined_outputs = torch.cat([second_output, first_output])
        
        return i.clone(), t.clone(), combined_outputs
    
    def inverse(self, i, t, latent_pair):
        # Inverse method for memcnn
        output = self.forward(i, t, latent_pair, reverse=True)
        return output