from tqdm import tqdm
from einops import rearrange
import numpy as np
import torch
from torchvision import transforms
import glob
import torch
import torchvision
from torchvision.io import VideoReader
torchvision.set_video_backend("video_reader")
import os
# from ..dists import frechet_distance
import itertools
from scipy import linalg

def load_video_clip_from_videoreader(
		av_reader,
		clip_start_timestamp,
		clip_duration,
		video_fps,
		video_num_frame,
		image_size,
		flip=False,
		randcrop=False,
		normalize=False
):
	av_reader.set_current_stream("video")
	keyframe_coverage = 1. / video_fps
	
	video_frames = []
	frame_timestamp = clip_start_timestamp
	for i, frame in enumerate(itertools.takewhile(
			lambda x: x['pts'] <= clip_start_timestamp + clip_duration + keyframe_coverage / 2.,
			av_reader.seek(max(clip_start_timestamp, 0.))
	)):
		if frame["pts"] >= frame_timestamp:
			video_frames.append(frame["data"])  # (c, h, w) tensor [0, 255]
			frame_timestamp += keyframe_coverage
		
		if len(video_frames) == video_num_frame:
			break
	
	if len(video_frames) < video_num_frame:
		res_length = video_num_frame - len(video_frames)
		for _ in range(res_length):
			video_frames.append(video_frames[-1])
	
	video_frames = torch.stack(video_frames, dim=0).float() / 255.
	
	# video_frames = load_and_transform_images_stable_diffusion(
	# 	video_frames,
	# 	size=image_size,
	# 	flip=False,
	# 	randcrop=False,
	# 	normalize=normalize
	# ).float()  # (n_frame, 3, h, w) in range [0., 1.]
	
	return video_frames


def load_av_clips_uniformly(
		video_path: str,
		video_fps: int = 8,
		video_num_frame: int = 16,
		image_size  = 256,
		num_clips: int = 1,
		load_audio_as_melspectrogram: bool = False,
):
	'''
	Return:
		video_frames: (b f c h w) in [0, 1]
		audio_frames:
			if load_audio_as_melspectrogram is True: (b 1 n t)
			else: List of tensors (b c ti), ti can be different
	'''
	# print(">>image_sizeL: ", image_size, video_num_frame)
	clip_duration = video_num_frame / video_fps
	av_reader = VideoReader(video_path, stream="video")
	meta_data = av_reader.get_metadata()
	video_duration, orig_video_fps = float(meta_data["video"]["duration"][0]), float(meta_data["video"]["fps"][0])
	# audio_duration, audio_sr = float(meta_data["audio"]["duration"][0]), int(meta_data["audio"]["framerate"][0])
	av_duration = video_duration #min(video_duration, audio_duration)
	# assert av_duration >= clip_duration, [video_path, video_duration, audio_duration]
	
	# 1. Sample clip start times
	if num_clips == 1:
		clip_start_timestamps = np.array([(av_duration - clip_duration) / 2.])
	else:
		clip_start_timestamps = np.linspace(0., av_duration - clip_duration, endpoint=True, num=num_clips)
	
	video_frames = []
	audio_frames = []
	for clip_start_timestamp in clip_start_timestamps:
		video_frames.append(
			load_video_clip_from_videoreader(
				av_reader,
				clip_start_timestamp,
				clip_duration,
				video_fps,
				video_num_frame,
				image_size,
				flip=False,
				randcrop=False,
				normalize=False
			)
		)
		# audio_frames.append(
		# 	load_audio_clip_from_videoreader(
		# 		av_reader,
		# 		clip_start_timestamp,
		# 		clip_duration,
		# 		audio_sr,
		# 		load_audio_as_melspectrogram
		# 	)
		# )
	
	video_frames = torch.stack(video_frames)  # (b, t, c, h, w)
	# if load_audio_as_melspectrogram:
	# 	audio_frames = torch.stack(audio_frames)  # (b, 1, c, t)
	
	return video_frames, audio_frames

def frechet_distance(x1, x2, eps=1e-6):
	"""Numpy implementation of the Frechet Distance.
	The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
	and X_2 ~ N(mu_2, C_2) is
			d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

	Stable version by Dougal J. Sutherland.

	Params:
	-- mu1   : Numpy array containing the activations of a layer of the
			   inception net (like returned by the function 'get_predictions')
			   for generated samples.
	-- mu2   : The sample mean over activations, precalculated on an
			   representative data set.
	-- sigma1: The covariance matrix over activations for generated samples.
	-- sigma2: The covariance matrix over activations, precalculated on an
			   representative data set.

	Returns:
	--   : The Frechet Distance.
	"""
	
	x1 = x1.numpy()
	x2 = x2.numpy()
	
	mu1 = np.mean(x1, axis=0)
	sigma1 = np.cov(x1, rowvar=False)
	
	mu2 = np.mean(x2, axis=0)
	sigma2 = np.cov(x2, rowvar=False)

	mu1 = np.atleast_1d(mu1)
	mu2 = np.atleast_1d(mu2)

	sigma1 = np.atleast_2d(sigma1)
	sigma2 = np.atleast_2d(sigma2)

	assert mu1.shape == mu2.shape, \
		'Training and test mean vectors have different lengths'
	assert sigma1.shape == sigma2.shape, \
		'Training and test covariances have different dimensions'

	diff = mu1 - mu2

	# Product might be almost singular
	covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
	if not np.isfinite(covmean).all():
		msg = ('fid calculation produces singular product; '
			   'adding %s to diagonal of cov estimates') % eps
		print(msg)
		offset = np.eye(sigma1.shape[0]) * eps
		covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

	# Numerical error might give slight imaginary component
	if np.iscomplexobj(covmean):
		if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
			m = np.max(np.abs(covmean.imag))
			raise ValueError('Imaginary component {}'.format(m))
		covmean = covmean.real

	tr_covmean = np.trace(covmean)

	return (diff.dot(diff) + np.trace(sigma1)
			+ np.trace(sigma2) - 2 * tr_covmean)

_I3D_PRETRAINED_ID = '1mQK8KD8G6UWRa5t87SRMm5PVXtlpneJT'
def load_i3d_pretrained(device=torch.device('cpu')):
	filepath = os.path.join("pretrained", 'i3d_torchscript.pt')
	if not os.path.exists(filepath):
		print("please download https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt")
		# os.system(f"wget {i3D_WEIGHTS_URL} -O {filepath}")
	i3d = torch.jit.load(filepath).eval().to(device)
	i3d.eval()
	return i3d

def preprocess_videos(videos, sequence_length=None):
	# video: BCTHW, [0, 1]
	assert videos.ndim == 5
	b, c, t, h, w = videos.shape
	# temporal crop
	if sequence_length is not None:
		assert sequence_length <= t
		videos = videos[:, :, :sequence_length]
		videos = videos.contiguous()
		t = sequence_length
	
	videos = rearrange(videos, "b c t h w -> (b t) c h w")
	
	transform_func = transforms.Compose([
		transforms.Resize(
			(224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
		),
		transforms.CenterCrop(224)
	])
	videos = transform_func(videos)
	videos = videos * 2 - 1
	
	videos = rearrange(videos, "(b t) c h w -> b c t h w", t=t)
	
	return videos

@torch.no_grad()
def compute_fvd_video_features(videos, net):
	'''
		videos in shape BCTHW in [0., 1.]
	'''
	videos = preprocess_videos(videos).contiguous()
	
	detector_kwargs = dict(rescale=False, resize=False, return_features=True)
	logits = net(videos, **detector_kwargs)
	# logits = net(videos)
	return logits

def read_videos(video_paths):
	fvd_features_arr = []
	
	for video_path in video_paths:
		video_tensors, audio_tensors = load_av_clips_uniformly(
				video_path=video_path,
				video_fps=video_fps,
				video_num_frame=video_num_frame,
				image_size=image_size,
				num_clips=1,
				load_audio_as_melspectrogram=True
			)


        # fvd_features = compute_fvd_video_features(
        #                 rearrange(video_tensors, "b f c h w -> b c f h w"),
        #                 i3d
        #             ).detach().cpu() # (b, c)
        # generated_fvd_features.append(fvd_features)    
		video_tensors= video_tensors.to(dtype=torch.float16, device="cuda:0")
		# print(">> video_tensors: ", video_tensors.shape)
		if video_tensors.shape[1] > 16:
			video_tensors = video_tensors[:, ::2]
		fvd_features = compute_fvd_video_features(
                    rearrange(video_tensors, "b f c h w -> b c f h w"),
                    i3d
                ).detach().cpu() # (b, c)
		fvd_features_arr.append(fvd_features)
        # break

	return fvd_features_arr

def find_intersect(source, target, gt_video_root, generated_video_root):
	new_source = []
	new_target = []
	idd = source[0].split("/")[-1]
	source_path = gt_video_root #source[0][:len(idd)]
	print(">> source_pathsource_path: ", source_path)
	for ss in target:
		idd = ss.split("/")[-1]
		new_selected_file = source_path+"/"+idd #+ ".mp4"
		# print(">>new_selected_file: ", new_selected_file)
		new_source.append(new_selected_file)

	print("LEN : ", len(new_source))
	return new_source, target



#### more iters: 
#### animate diff titanguide-eps results: 50iters: , 100 iters:
#### animate diff titanguide-gradest results: 50iters: , 100 iters:

if __name__== "__main__":
	device = "cuda:0"
	dtype = torch.float16
	i3d = load_i3d_pretrained(device).to(device=device, dtype=dtype)

	### inpaint fvd: freedom: 170.7727428661965, 176 (noise),  mpgd: 169.879, tfg: 
    # gt_video_root="/mnt/data2/chris/datasets/vggsound/reference_videos_2secs/"
	gt_video_root="/mnt/data2/chris/datasets/vggsound/reference_videos_2secs/"
	## noise: 211.838, scorebased: 
	generated_video_root = "/mnt/data2/chris/outputs/video_gen_guide/imagebind/animatediff/titan_gradest_imagebind_100iters/" #"/mnt/data2/chris/outputs/video_gen_guide/imagebind/cogvid/tfg_imagebind"
	# generated_video_root="/mnt/data2/chris/outputs/video_gen_guide/imagebind/cogvid/titan_imagebind_guide_vanilla"
	# generated_video_root="/mnt/data2/chris/outputs/video_gen_guide/imagebind/titan_vanilla_imagebind_50iters"
	# generated_video_root = "/mnt/data2/chris/outputs/video_gen_guide/imagebind/titan_moreiters_100"
	# generated_video_root="/mnt/data2/chris/outputs/video_gen_guide/imagebind/titan256x256_gradest_part3/"
	# generated_video_root= '/mnt/data2/chris/outputs/video_gen_guide/video_inpaint/titan_scorebased/'# generated_video_root= '/mnt/data2/chris/outputs/video_gen_guide/imagebind/titan384x384eps_bener/'
	# generated_video_root= "/mnt/data2/chris/outputs/video_gen_guide/video_inpaint/tfg/" #"/mnt/data2/chris/outputs/vgg/test_seeingandhearing_textupd/" #"/mnt/data2/chris/outputs/vgg/test_ours_epsdiff/" ##"/mnt/data2/chris/outputs/vgg/test_seeinghearing/"
	
	gt_video_paths = glob.glob(f"{gt_video_root}/*.mp4")
	generated_video_paths = glob.glob(f"{generated_video_root}/*.mp4")
	# gt_video_paths, generated_video_paths = find_intersect(gt_video_paths, generated_video_paths, gt_video_root, generated_video_root)

	image_size = (256, 256) #(256, 256) ## 384x384: FVD 353
	video_fps= 8
	video_num_frame = 16
	print(">>gt_video_paths: ", len(gt_video_paths), len(generated_video_paths))
	groundtruth_fvd_features = read_videos(gt_video_paths)


	#### COGVIDEO ONLY, 8,16 for others
	video_fps= 8
	video_num_frame = 32
	generated_fvd_features = read_videos(generated_video_paths)

	groundtruth_fvd_features = torch.cat(groundtruth_fvd_features)
	generated_fvd_features = torch.cat(generated_fvd_features)
	fvd_score = frechet_distance(groundtruth_fvd_features, generated_fvd_features)

	print("FVD: ", fvd_score) ## part 0 gradest : 449.92, part 0 50 iternoise: 447.35  #.52 #406.78