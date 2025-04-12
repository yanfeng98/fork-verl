<h1 style="text-align: center;">verl: Volcano Engine Reinforcement Learning for LLMs</h1>

verl is a flexible, efficient and production-ready RL training library for large language models (LLMs).

verl is the open-source version of **[HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)** paper.

verl is flexible and easy to use with:

- **Easy extension of diverse RL algorithms**: The hybrid-controller programming model enables flexible representation and efficient execution of complex Post-Training dataflows. Build RL dataflows such as GRPO, PPO in a few lines of code.

- **Seamless integration of existing LLM infra with modular APIs**: Decouples computation and data dependencies, enabling seamless integration with existing LLM frameworks, such as FSDP, Megatron-LM, vLLM, SGLang, etc

- **Flexible device mapping**: Supports various placement of models onto different sets of GPUs for efficient resource utilization and scalability across different cluster sizes.

- Ready integration with popular HuggingFace models


verl is fast with:

- **State-of-the-art throughput**: SOTA LLM training and inference engine integrations and SOTA RL throughput.

- **Efficient actor model resharding with 3D-HybridEngine**: Eliminates memory redundancy and significantly reduces communication overhead during transitions between training and generation phases.

</p>

## News
- [2025/03] [DAPO](https://dapo-sia.github.io/) is the open-sourced SOTA RL algorithm that achieves 50 points on AIME 2024 based on the Qwen2.5-32B pre-trained model, surpassing the previous SOTA achieved by DeepSeek's GRPO (DeepSeek-R1-Zero-Qwen-32B). DAPO's training is fully powered by verl and the reproduction code is [publicly available](https://github.com/volcengine/verl/tree/gm-tyx/puffin/main/recipe/dapo) now.

## Key Features

- **FSDP** and **Megatron-LM** for training.
- **vLLM**, **SGLang**(experimental) and **HF Transformers** for rollout generation.
- Compatible with Hugging Face Transformers and Modelscope Hub: Qwen-2.5, Llama3.1, Gemma2, DeepSeek-LLM, etc
- Supervised fine-tuning.
- Reinforcement learning with [PPO](examples/ppo_trainer/), [GRPO](examples/grpo_trainer/), [ReMax](examples/remax_trainer/), [REINFORCE++](https://verl.readthedocs.io/en/latest/examples/config.html#algorithm), [RLOO](examples/rloo_trainer/), [PRIME](recipe/prime/), etc.
  - Support model-based reward and function-based reward (verifiable reward)
  - Support vision-language models (VLMs) and [multi-modal RL](examples/grpo_trainer/run_qwen2_5_vl-7b.sh)
- Flash attention 2, [sequence packing](examples/ppo_trainer/run_qwen2-7b_seq_balance.sh), [sequence parallelism](examples/ppo_trainer/run_deepseek7b_llm_sp2.sh) support via DeepSpeed Ulysses, [LoRA](examples/sft/gsm8k/run_qwen_05_peft.sh), [Liger-kernel](examples/sft/gsm8k/run_qwen_05_sp2_liger.sh).
- Scales up to 70B models and hundreds of GPUs.
- Experiment tracking with wandb, swanlab, mlflow and tensorboard.

## Getting Started

<a href="https://verl.readthedocs.io/en/latest/index.html"><b>Documentation</b></a>

## Installation

```bash
# python 12
python -m venv env
source env/bin/activate
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
# torch 2.6
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
# https://github.com/Dao-AILab/flash-attention/releases
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**Quickstart:**
- [Installation](https://verl.readthedocs.io/en/latest/start/install.html)
- [Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)
- [Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html)

**Running a PPO example step-by-step:**
- Data and Reward Preparation
  - [Prepare Data for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)
  - [Implement Reward Function for Dataset](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)
- Understanding the PPO Example
  - [PPO Example Architecture](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html)
  - [Config Explanation](https://verl.readthedocs.io/en/latest/examples/config.html)
  - [Run GSM8K Example](https://verl.readthedocs.io/en/latest/examples/gsm8k_example.html)

**Reproducible algorithm baselines:**
- [PPO, GRPO, ReMax](https://verl.readthedocs.io/en/latest/experiment/ppo.html)

**For code explanation and advance usage (extension):**
- PPO Trainer and Workers
  - [PPO Ray Trainer](https://verl.readthedocs.io/en/latest/workers/ray_trainer.html)
  - [PyTorch FSDP Backend](https://verl.readthedocs.io/en/latest/workers/fsdp_workers.html)
  - [Megatron-LM Backend](https://verl.readthedocs.io/en/latest/index.html)
- Advance Usage and Extension
  - [Ray API design tutorial](https://verl.readthedocs.io/en/latest/advance/placement.html)
  - [Extend to Other RL(HF) algorithms](https://verl.readthedocs.io/en/latest/advance/dpo_extension.html)
  - [Add Models with the FSDP Backend](https://verl.readthedocs.io/en/latest/advance/fsdp_extension.html)
  - [Add Models with the Megatron-LM Backend](https://verl.readthedocs.io/en/latest/advance/megatron_extension.html)
  - [Deployment using Separate GPU Resources](https://github.com/volcengine/verl/tree/main/examples/split_placement)

## Performance Tuning Guide
The performance is essential for on-policy RL algorithm. We have written a detailed [performance tuning guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html) to help you optimize performance.

## Use vLLM v0.8.2
veRL now supports vLLM>=0.8.2 when using FSDP as the training backend. Please refer to [this document](https://github.com/volcengine/verl/blob/main/docs/README_vllm0.8.md) for installation guide and more information. Please avoid vllm 0.7.x which contains bugs that may lead to OOMs and unexpected errors.

## Awesome work using verl
- [TinyZero](https://github.com/Jiayi-Pan/TinyZero): a reproduction of **DeepSeek R1 Zero** recipe for reasoning tasks ![GitHub Repo stars](https://img.shields.io/github/stars/Jiayi-Pan/TinyZero)
- [DAPO](https://dapo-sia.github.io/): the fully open source SOTA RL algorithm that beats DeepSeek-R1-zero-32B ![GitHub Repo stars](https://img.shields.io/github/stars/volcengine/verl)
- [SkyThought](https://github.com/NovaSky-AI/SkyThought): RL training for Sky-T1-7B by NovaSky AI team. ![GitHub Repo stars](https://img.shields.io/github/stars/NovaSky-AI/SkyThought)
- [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason): SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild ![GitHub Repo stars](https://img.shields.io/github/stars/hkust-nlp/simpleRL-reason)
- [Easy-R1](https://github.com/hiyouga/EasyR1): **Multi-modal** RL training framework ![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/EasyR1)
- [OpenManus-RL](https://github.com/OpenManus/OpenManus-RL): LLM Agents RL tunning framework for multiple agent environments. ![GitHub Repo stars](https://img.shields.io/github/stars/OpenManus/OpenManus-RL)
- [deepscaler](https://github.com/agentica-project/deepscaler): iterative context scaling with GRPO ![GitHub Repo stars](https://img.shields.io/github/stars/agentica-project/deepscaler)
- [PRIME](https://github.com/PRIME-RL/PRIME): Process reinforcement through implicit rewards ![GitHub Repo stars](https://img.shields.io/github/stars/PRIME-RL/PRIME)
- [RAGEN](https://github.com/ZihanWang314/ragen): a general-purpose reasoning **agent** training framework ![GitHub Repo stars](https://img.shields.io/github/stars/ZihanWang314/ragen)
- [Logic-RL](https://github.com/Unakar/Logic-RL): a reproduction of DeepSeek R1 Zero on 2K Tiny Logic Puzzle Dataset. ![GitHub Repo stars](https://img.shields.io/github/stars/Unakar/Logic-RL)
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1): RL with reasoning and **searching (tool-call)** interleaved LLMs ![GitHub Repo stars](https://img.shields.io/github/stars/PeterGriffinJin/Search-R1)
- [ReSearch](https://github.com/Agent-RL/ReSearch): Learning to **Re**ason with **Search** for LLMs via Reinforcement Learning ![GitHub Repo stars](https://img.shields.io/github/stars/Agent-RL/ReSearch)
- [DeepRetrieval](https://github.com/pat-jj/DeepRetrieval): Hacking **Real Search Engines** and **retrievers** with LLMs via RL for **information retrieval** ![GitHub Repo stars](https://img.shields.io/github/stars/pat-jj/DeepRetrieval)
- [Code-R1](https://github.com/ganler/code-r1): Reproducing R1 for **Code** with Reliable Rewards ![GitHub Repo stars](https://img.shields.io/github/stars/ganler/code-r1)
- [cognitive-behaviors](https://github.com/kanishkg/cognitive-behaviors): Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs ![GitHub Repo stars](https://img.shields.io/github/stars/kanishkg/cognitive-behaviors)
- [PURE](https://github.com/CJReinforce/PURE): **Credit assignment** is the key to successful reinforcement fine-tuning using **process reward model** ![GitHub Repo stars](https://img.shields.io/github/stars/CJReinforce/PURE)
- [MetaSpatial](https://github.com/PzySeere/MetaSpatial): Reinforcing **3D Spatial Reasoning** in **VLMs** for the **Metaverse** ![GitHub Repo stars](https://img.shields.io/github/stars/PzySeere/MetaSpatial)
- [DeepEnlighten](https://github.com/DolbyUUU/DeepEnlighten): Reproduce R1 with **social reasoning** tasks and analyze key findings ![GitHub Repo stars](https://img.shields.io/github/stars/DolbyUUU/DeepEnlighten)
- [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher): Scaling deep research via reinforcement learning in real-world environments ![GitHub Repo stars](https://img.shields.io/github/stars/GAIR-NLP/DeepResearcher)
- [self-rewarding-reasoning-LLM](https://arxiv.org/pdf/2502.19613): self-rewarding and correction with **generative reward models** ![GitHub Repo stars](https://img.shields.io/github/stars/RLHFlow/Self-rewarding-reasoning-LLM)
- [critic-rl](https://github.com/HKUNLP/critic-rl): LLM critics for code generation ![GitHub Repo stars](https://img.shields.io/github/stars/HKUNLP/critic-rl)
- [VAGEN](https://github.com/RAGEN-AI/VAGEN): Training VLM agents with multi-turn reinforcement learning ![GitHub Repo stars](https://img.shields.io/github/stars/RAGEN-AI/VAGEN)
- [DQO](https://arxiv.org/abs/2410.09302): Enhancing multi-Step reasoning abilities of language models through direct Q-function optimization
- [FIRE](https://arxiv.org/abs/2410.21236): Flaming-hot initiation with regular execution sampling for large language models
- [Rec-R1](https://arxiv.org/pdf/2503.24289): Bridging Generative Large Language Models and Recommendation Systems via Reinforcement Learning
- [all-hands/openhands-lm-32b-v0.1](https://www.all-hands.dev/blog/introducing-openhands-lm-32b----a-strong-open-coding-agent-model): A strong, open coding agent model, trained with [multi-turn fine-tuning](https://github.com/volcengine/verl/pull/195)
- [AdaRFT](https://github.com/uscnlp-lime/verl): Efficient Reinforcement Finetuning via **Adaptive Curriculum Learning** ![GitHub Repo stars](https://img.shields.io/github/stars/uscnlp-lime/verl)
- [Trust Region Preference Approximation](https://github.com/XueruiSu/Trust-Region-Preference-Approximation): A simple and stable **reinforcement learning algorithm** for LLM reasoning.  ![GitHub Repo stars](https://img.shields.io/github/stars/XueruiSu/Trust-Region-Preference-Approximation)

## Code formatting
We use yapf (Google style) to enforce strict code formatting when reviewing PRs. To reformat your code locally, make sure you have installed the **latest** version of `yapf`
```bash
pip3 install yapf --upgrade
```
Then, make sure you are at top level of verl repo and run
```bash
bash scripts/format.sh
```
