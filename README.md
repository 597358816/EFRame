

**EFRame** is a novel augmentation of Group Relative Policy Optimization (GRPO) that systematically addresses the exploration, stability, and sample-efficiency bottlenecks in RL-based reasoning for LLMs. By introducing three complementary modules—Exploration, Filtering, and Replay—EFRame creates a closed-loop learning cycle that guides the model from broad exploration to stable convergence and deep knowledge acquisition.

## Features

- **Core Algorithm**  
  - Exploration-Filtering-Replay augmentation of GRPO  
  - Seamless plug-in to any GRPO-compatible training pipeline

- **Key Modules**  
  1. **Exploration**: additional rollouts to cover high-quality trajectories  
  2. **Filtering**: online sample selection to discard low-quality/noisy experiences  
  3. **Replay**: prioritized experience replay to reinforce rare but informative samples  

- **Benefits**  
  - Improved exploration and coverage of the state space  
  - Reduced variance and greater training stability  
  - Higher sample efficiency and faster convergence  
  - Enables fine-grained analysis of sample contributions to learning  

## Requirements

### Software

Install via pip:

```bash
pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 vllm==0.8.3 transformers==4.51.2
pip install flash-attn
pip install -e .
pip install tensorboard
