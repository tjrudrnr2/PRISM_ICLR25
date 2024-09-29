# PRISM: PRivacy-preserving Improved Stochastic Masking for federated generative models
## Requirements
```
conda create -n prism python=3.10.9
conda activate prism
pip install -r requirements.txt
pip3 install torch torchvision torchaudio
```
## RUN
```
wandb login PERSONAL_API_KEY
```
- --num_scorelayer=0~1, for $\textbf{PRISM}-\alpha$. Default configuration is 0 ($\textbf{PRISM}$).
- --save=True is an optional, if you want to log WANDB.

## IID case
```
python run_prism.py --model=prism --aggregation=BA --dataset=mnist --iid=1 --gpunum=0 --epochs=300 --experiments=EXPERIMENTS --dp_epsilon=9.8 --server_ema --num_scorelayer=0
```
## Non-IID case
```
python run_prism.py --model=prism --aggregation=BA --dataset=mnist --iid=0 --divide=4 --gpunum=0 --epochs=150 --experiments=EXPERIMENTS --dp_epsilon=9.8 --server_ema --num_scorelayer=0 
```
