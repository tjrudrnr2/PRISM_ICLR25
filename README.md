# PRISM: PRivacy-preserving Improved Stochastic Masking for federated generative models
## Requirements
```
conda create -n prism python=3.10.9
conda activate prism
pip install -r requirements.txt
pip3 install torch torchvision torchaudio
```
```
wandb login PERSONAL_API_KEY
```
- --save=True is an optional, if you want to log WANDB.

# RUN
## IID
- DP case
```
python ./run_prism.py --model=prism --aggregation=BA --dataset=mnist --gpu=0 --iid=1 --gpunum=0 --num_scorelayer=0 --epochs=500 --experiments=EXRIMENTS --dp_epsilon=9.8 --dynamic_ema --save=True
```
- No-DP case
```
python ./run_prism.py --model=prism --aggregation=BA --dataset=mnist --gpu=0 --iid=1 --gpunum=0 --num_scorelayer=0 --epochs=500 --experiments=EXRIMENTS --dynamic_ema --save=True
```
## Non-IID case
- DP case
```
python ./run_prism.py --model=prism --aggregation=BA --dataset=mnist --gpu=0 --iid=0 --divide=4 --gpunum=0 --num_scorelayer=0 --epochs=500 --experiments=EXRIMENTS --dp_epsilon=9.8 --dynamic_ema --save=True
```
- No-DP case
```
python ./run_prism.py --model=prism --aggregation=BA --dataset=mnist --gpu=0 --iid=0 --divide=4 --gpunum=0 --num_scorelayer=0 --epochs=500 --experiments=EXRIMENTS --dynamic_ema --save=True
```
## PRISM-$\alpha$
- --num_scorelayer=0~1, for $\textbf{PRISM}-\alpha$. Default configuration is 0 ($\textbf{PRISM}$).
```
python ./run_prism.py --model=prism --aggregation=BA --dataset=mnist --gpu=0 --iid=1 --gpunum=0 --num_scorelayer=0 --epochs=500 --experiments=EXRIMENTS --dp_epsilon=9.8 --num_scorelayer=0.8 --save=True
```