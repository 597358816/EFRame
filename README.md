## Requirements

### Software

Install via pip:

```bash
conda create -n UEC-RL python=3.11
conda activate UEC-RL

pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 vllm==0.8.3 transformers==4.51.2 
pip install ray==2.48.0 tensordict==0.9.1 pydantic==2.11.7
pip install flash-attn
pip install -e .
pip install tensorboard
cd example
bash qwen-vl-7b.sh
