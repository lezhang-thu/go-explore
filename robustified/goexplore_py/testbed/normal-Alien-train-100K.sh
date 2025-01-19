set -ex

#for i in 23 29 31 37 41 43 47 53 59 61
for i in 23
do
CUDA_VISIBLE_DEVICES=0 python normal_ppo.py \
	--seed $i --env PongNoFrameskip-v4 \
	--epochs 10
done
