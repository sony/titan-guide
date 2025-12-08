CUDA_VISIBLE_DEVICES=2
data_type=text2videostd
# data_type=text2image
image_size=256
dataset="parti_prompts"
model_name_or_path='emilianJR/epiCRealism'
save_path='outputs/style_transfer/titan256x256_2'
task=style_transfer_nograd
guide_network='openai/clip-vit-base-patch16'
target='data/wikiart/0.png'
recur_steps=1 
rho_schedule="decrease"
 
train_steps=1000
inference_steps=15 #15 #20 #15 #15
eta=1.0
clip_x0=False
seed=42
logging_dir='logs'
per_sample_batch_size=1
num_samples=1100
logging_resolution=512
guidance_name='titanguideest'
# guidance_name='freedomori'
eval_batch_size=1
wandb=False
guidance_strength=0.18 #0.80522057100290275
# guidance_strength=1. #0.001 #40 #1.78 #0.072761
# guidance_strength=.1 #75 #0.8 #2.15 #.85 #2.5

# rho=0.25
rho=.5
mu=2
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
    --rho_schedule $rho_schedule \
    --recur_steps $recur_steps \
    --save_path $save_path \
    --eval_batch_size $eval_batch_size


