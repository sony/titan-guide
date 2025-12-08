CUDA_VISIBLE_DEVICES=0
data_type=text2videocogvid16f
image_size=512
dataset="parti_prompts"
model_name_or_path='emilianJR/epiCRealism'


task=style_transfer
guide_network='openai/clip-vit-base-patch16'
target='/mnt/data2/chris/code/Training-Free-Guidance/data/wikiart/0.png'
recur_steps=1
rho_schedule="decrease"
save_path="/mnt/data2/chris/outputs/video_gen_guide/style_transfer/tfg_cogvid/"


train_steps=1000
inference_steps=20 #15
eta=1.0
clip_x0=False
seed=42
logging_dir='logs'
per_sample_batch_size=1
num_samples=1050
logging_resolution=512
guidance_name='tfg_video'
eval_batch_size=1
wandb=False
guidance_strength=.5 #

# rho=0.25
rho=.5
mu=2 #2
sigma=0.1
eps_bsz=1

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py \
    --data_type $data_type \
    --task $task \
    --image_size $image_size \
    --dataset $dataset \
    --guide_network $guide_network \
    --logging_resolution $logging_resolution \
    --model_name_or_path $model_name_or_path \
    --train_steps $train_steps \
    --inference_steps $inference_steps \
    --target $target \
    --eta $eta \
    --clip_x0 $clip_x0 \
    --rho $rho \
    --mu $mu \
    --sigma $sigma \
    --eps_bsz $eps_bsz \
    --wandb $wandb \
    --seed $seed \
    --logging_dir $logging_dir \
    --per_sample_batch_size $per_sample_batch_size \
    --num_samples $num_samples \
    --guidance_name $guidance_name \
    --guidance_strength $guidance_strength \
    --recur_steps $recur_steps \
    --save_path $save_path
    --eval_batch_size $eval_batch_size


