<div align="center">

<a href="https://deepwiki.com/volcengine/verl"><img src="https://devin.ai/assets/deepwiki-badge.png" alt="Ask DeepWiki.com" style="height:20px;"></a>
[![Documentation](https://img.shields.io/badge/documentation-blue)](https://verl.readthedocs.io/en/latest/)

</div>

<h1 style="text-align: center;">verl: Volcano Engine Reinforcement Learning for LLMs</h1>

verl is a flexible, efficient and production-ready RL training library for large language models (LLMs).

verl is flexible and easy to use with:

- **Easy extension of diverse RL algorithms**: The hybrid-controller programming model enables flexible representation and efficient execution of complex post-training dataflows. Build RL dataflows such as GRPO, PPO in a few lines of code.

- **Seamless integration of existing LLM infra with modular APIs**: Decouples computation and data dependencies, enabling seamless integration with existing LLM frameworks, such as FSDP, Megatron-LM, vLLM, SGLang, etc

- **Flexible device mapping**: Supports various placement of models onto different sets of GPUs for efficient resource utilization and scalability across different cluster sizes.

- Ready integration with popular HuggingFace models

verl is fast with:

- **State-of-the-art throughput**: SOTA LLM training and inference engine integrations and SOTA RL throughput.

- **Efficient actor model resharding with 3D-HybridEngine**: Eliminates memory redundancy and significantly reduces communication overhead during transitions between training and generation phases.

</p>

## Key Features

- **FSDP**, **FSDP2** and **Megatron-LM** for training.
- **vLLM**, **SGLang** and **HF Transformers** for rollout generation.
- Compatible with Hugging Face Transformers and Modelscope Hub: [Qwen-3](https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen3-8b.sh), Qwen-2.5, Llama3.1, Gemma2, DeepSeek-LLM, etc
- Supervised fine-tuning.
- Reinforcement learning with [PPO](examples/ppo_trainer/), [GRPO](examples/grpo_trainer/), [GSPO](recipe/gspo/), [ReMax](examples/remax_trainer/), [REINFORCE++](https://verl.readthedocs.io/en/latest/examples/config.html#algorithm), [RLOO](examples/rloo_trainer/), [PRIME](recipe/prime/), [DAPO](recipe/dapo/), [DrGRPO](recipe/drgrpo), [KL_Cov & Clip_Cov](recipe/entropy) etc.
  - Support model-based reward and function-based reward (verifiable reward) for math, [coding](https://github.com/volcengine/verl/tree/main/recipe/dapo), etc
  - Support vision-language models (VLMs) and [multi-modal RL](examples/grpo_trainer/run_qwen2_5_vl-7b.sh) with Qwen2.5-vl, Kimi-VL
  - [Multi-turn with tool calling](https://github.com/volcengine/verl/tree/main/examples/sglang_multiturn)
- LLM alignment recipes such as [Self-play preference optimization (SPPO)](https://github.com/volcengine/verl/tree/main/recipe/sppo)
- Flash attention 2, [sequence packing](examples/ppo_trainer/run_qwen2-7b_seq_balance.sh), [sequence parallelism](examples/ppo_trainer/run_deepseek7b_llm_sp2.sh) support via DeepSpeed Ulysses, [LoRA](examples/sft/gsm8k/run_qwen_05_peft.sh), [Liger-kernel](examples/sft/gsm8k/run_qwen_05_sp2_liger.sh).
- Scales up to 671B models and hundreds of GPUs with [expert parallelism](https://github.com/volcengine/verl/pull/1467)
- Multi-gpu [LoRA RL](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html) support to save memory.
- Experiment tracking with wandb, swanlab, mlflow and tensorboard.

## Getting Started

<a href="https://verl.readthedocs.io/en/latest/index.html"><b>Documentation</b></a>

**Quickstart:**

- [Installation](./docs/start/install.rst)
- [Quickstart](./docs/start/quickstart.rst)
- [Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html) & [Tech Talk](https://hcqnc.xetlk.com/sl/3vACOK) (in Chinese)
- [PPO in verl](https://verl.readthedocs.io/en/latest/algo/ppo.html)
- [GRPO in verl](./docs/algo/grpo.md)

**Running a PPO example step-by-step:**

- [Prepare Data for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)
- [Implement Reward Function for Dataset](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)
- [PPO Example Architecture](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html)
- [Config Explanation](https://verl.readthedocs.io/en/latest/examples/config.html)

**Reproducible algorithm baselines:**

- [RL performance on coding, math](https://verl.readthedocs.io/en/latest/algo/baseline.html)

**For code explanation and advance usage (extension):**

- PPO Trainer and Workers
  - [PPO Ray Trainer](https://verl.readthedocs.io/en/latest/workers/ray_trainer.html)
  - [PyTorch FSDP Backend](https://verl.readthedocs.io/en/latest/workers/fsdp_workers.html)
  - [Megatron-LM Backend](https://verl.readthedocs.io/en/latest/index.html)

- Advanced Usage and Extension
  - [Add Models with the FSDP Backend](https://verl.readthedocs.io/en/latest/advance/fsdp_extension.html)
  - [Add Models with the Megatron-LM Backend](https://verl.readthedocs.io/en/latest/advance/megatron_extension.html)
  - [Multi-turn Rollout Support](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)
  - [Search Tool Integration](https://verl.readthedocs.io/en/latest/sglang_multiturn/search_tool_example.html)
  - [Sandbox Fusion Integration](https://verl.readthedocs.io/en/latest/examples/sandbox_fusion_example.html)
  - [Deployment using Separate GPU Resources](https://github.com/volcengine/verl/tree/main/examples/split_placement)
  - [Extend to Other RL(HF) algorithms](https://verl.readthedocs.io/en/latest/advance/dpo_extension.html)
  - [Ray API design tutorial](https://verl.readthedocs.io/en/latest/advance/placement.html)

**Blogs from the community**

- [When Reasoning Models Break Tokenization: The Hidden Complexity of Multiturn Training](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/fast_tokenization/multiturn_tokenization_and_masking.md)
- [verl deployment on AWS SageMaker](https://medium.com/@kaige.yang0110/run-verl-on-sagemaker-using-4x8-l40s-gpus-8e6d5c3c61d3)
- [verl x SGLang Multi-turn Code Walkthrough](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/code-walk-through/readme_EN.md)
- [Optimizing SGLang Memory Usage in verl](https://hebiao064.github.io/rl-memory-management)
- [SGLang, verl, OpenBMB and Tsinghua University: Pioneering End-to-End Multi-Turn RLHF](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/verl-multiturn-rollout-Release.md)
- [Reinforcement Learning from Human Feedback on AMD GPUs with verl and ROCm Integration](https://rocm.blogs.amd.com/artificial-intelligence/verl-large-scale/README.html)
- [veMLP x verl ：玩转强化学习训练](https://mp.weixin.qq.com/s/7nbqxk4knMGd-hQE9ls2tA)
- [使用 verl 进行 GRPO 分布式强化学习训练最佳实践](https://www.volcengine.com/docs/6459/1463942)
- [HybridFlow verl 原文浅析](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/readme.md)
- [最高提升 20 倍吞吐量！豆包大模型团队发布全新 RLHF 框架，现已开源！](https://team.doubao.com/en/blog/%E6%9C%80%E9%AB%98%E6%8F%90%E5%8D%8720%E5%80%8D%E5%90%9E%E5%90%90%E9%87%8F-%E8%B1%86%E5%8C%85%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%9B%A2%E9%98%9F%E5%8F%91%E5%B8%83%E5%85%A8%E6%96%B0-rlhf-%E6%A1%86%E6%9E%B6-%E7%8E%B0%E5%B7%B2%E5%BC%80%E6%BA%90)

## Use Latest SGLang

SGLang is fully supported with verl, and SGLang RL Group is working extensively on building unique features, including multi-turn agentic RL, VLM RLHF, server-based RL, and partial rollout. Please refer to [this document](https://verl.readthedocs.io/en/latest/workers/sglang_worker.html) for the installation guide and more information.

## Upgrade to FSDP2

verl is fully embracing FSDP2! FSDP2 is recommended by torch distributed team, providing better throughput and memory usage, and is composible with other features (e.g. torch.compile). To enable FSDP2, simply use verl main and set the following options:
```
actor_rollout_ref.ref.strategy=fsdp2
actor_rollout_ref.actor.strategy=fsdp2
critic.strategy=fsdp2 
reward_model.strategy=fsdp2 
```
Furthermore, FSDP2 cpu offloading is compatible with gradient accumulation. You can turn it on to save memory with `actor_rollout_ref.actor.fsdp_config.offload_policy=True`. For more details, see https://github.com/volcengine/verl/pull/1026

## Awesome work using verl

- [TinyZero](https://github.com/Jiayi-Pan/TinyZero): a reproduction of **DeepSeek R1 Zero** recipe for reasoning tasks ![GitHub Repo stars](https://img.shields.io/github/stars/Jiayi-Pan/TinyZero)
- [SkyThought](https://github.com/NovaSky-AI/SkyThought): RL training for Sky-T1-7B by NovaSky AI team. ![GitHub Repo stars](https://img.shields.io/github/stars/NovaSky-AI/SkyThought)
- [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason): SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild ![GitHub Repo stars](https://img.shields.io/github/stars/hkust-nlp/simpleRL-reason)
- [Easy-R1](https://github.com/hiyouga/EasyR1): **Multi-modal** RL training framework ![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/EasyR1)
- [OpenManus-RL](https://github.com/OpenManus/OpenManus-RL): LLM Agents RL tunning framework for multiple agent environments. ![GitHub Repo stars](https://img.shields.io/github/stars/OpenManus/OpenManus-RL)
- [rllm](https://github.com/agentica-project/rllm): async RL training with [verl-pipeline](https://github.com/agentica-project/verl-pipeline) ![GitHub Repo stars](https://img.shields.io/github/stars/agentica-project/rllm)
- [RAGEN](https://github.com/ZihanWang314/ragen): a general-purpose reasoning **agent** training framework ![GitHub Repo stars](https://img.shields.io/github/stars/ZihanWang314/ragen)
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1): RL with reasoning and **searching (tool-call)** interleaved LLMs ![GitHub Repo stars](https://img.shields.io/github/stars/PeterGriffinJin/Search-R1)
- [ReSearch](https://github.com/Agent-RL/ReSearch): Learning to **Re**ason with **Search** for LLMs via Reinforcement Learning ![GitHub Repo stars](https://img.shields.io/github/stars/Agent-RL/ReSearch)
- [Skywork-OR1](https://github.com/SkyworkAI/Skywork-OR1): Skywork open reaonser series ![GitHub Repo stars](https://img.shields.io/github/stars/SkyworkAI/Skywork-OR1)
- [ToRL](https://github.com/GAIR-NLP/ToRL): Scaling tool-integrated RL ![GitHub Repo stars](https://img.shields.io/github/stars/GAIR-NLP/ToRL)
- [Absolute Zero Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner): [A no human curated data self-play framework for reasoning](https://arxiv.org/abs/2505.03335) ![GitHub Repo stars](https://img.shields.io/github/stars/LeapLabTHU/Absolute-Zero-Reasoner)
- [verl-agent](https://github.com/langfengQ/verl-agent): A scalable training framework for **long-horizon LLM/VLM agents**, along with a new algorithm **GiGPO** ![GitHub Repo stars](https://img.shields.io/github/stars/langfengQ/verl-agent)
- [RL-Factory](https://github.com/Simple-Efficient/RL-Factory): An easy and efficient RL post-training framework for Agentic Learning ![GitHub Repo stars](https://img.shields.io/github/stars/Simple-Efficient/RL-Factory)
- [ReTool](https://retool-rl.github.io/): ReTool: reinforcement learning for strategic tool use in LLMs. Code release is in progress...
- [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool): An unified and easy-to-extend tool-agent training framework based on verl![GitHub Repo stars](https://img.shields.io/github/stars/TIGER-AI-Lab/verl-tool)
- [PRIME](https://github.com/PRIME-RL/PRIME): Process reinforcement through implicit rewards ![GitHub Repo stars](https://img.shields.io/github/stars/PRIME-RL/PRIME)
- [MemAgent](https://github.com/BytedTsinghua-SIA/MemAgent): MemAgent: Reshaping Long-Context LLM with Multi-Conv RL based Memory Agent ![GitHub Repo stars](https://img.shields.io/github/stars/BytedTsinghua-SIA/MemAgent)
- [POLARIS](https://github.com/ChenxinAn-fdu/POLARIS): A Post-training recipe for scaling RL on Advanced Reasoning models ![GitHub Repo stars](https://img.shields.io/github/stars/ChenxinAn-fdu/POLARIS)
- [GUI-R1](https://github.com/ritzz-ai/GUI-R1): **GUI-R1**: A Generalist R1-style Vision-Language Action Model For **GUI Agents** ![GitHub Repo stars](https://img.shields.io/github/stars/ritzz-ai/GUI-R1)
- [DeepRetrieval](https://github.com/pat-jj/DeepRetrieval): RL Training of **Search Agent** with **Search/Retrieval Outcome** ![GitHub Repo stars](https://img.shields.io/github/stars/pat-jj/DeepRetrieval)
- [Code-R1](https://github.com/ganler/code-r1): Reproducing R1 for **Code** with Reliable Rewards ![GitHub Repo stars](https://img.shields.io/github/stars/ganler/code-r1)
- [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher): Scaling deep research via reinforcement learning in real-world environments ![GitHub Repo stars](https://img.shields.io/github/stars/GAIR-NLP/DeepResearcher)
- [VAGEN](https://github.com/RAGEN-AI/VAGEN): Training VLM agents with multi-turn reinforcement learning ![GitHub Repo stars](https://img.shields.io/github/stars/RAGEN-AI/VAGEN)
- [RM-R1](https://arxiv.org/abs/2505.02387): RL training of reasoning reward models ![GitHub Repo stars](https://img.shields.io/github/stars/RM-R1-UIUC/RM-R1)
- [LUFFY](https://arxiv.org/pdf/2504.14945): Learning to Reason under Off-Policy Guidance![GitHub Repo stars](https://img.shields.io/github/stars/ElliottYan/LUFFY)
- [DeepMath](https://github.com/zwhe99/DeepMath): DeepMath-103K data and series models for math reasoning![GitHub Repo stars](https://img.shields.io/github/stars/zwhe99/DeepMath)
- [PACS](https://github.com/ritzz-ai/PACS): Implicit Actor Critic Coupling via a Supervised Learning Framework for RLVR ![GitHub Repo stars](https://img.shields.io/github/stars/ritzz-ai/PACS)
- [Entropy Mechanism of RL](https://github.com/PRIME-RL/Entropy-Mechanism-of-RL): The Entropy Mechanism of Reinforcement Learning for Large Language Model Reasoning![GitHub Repo stars](https://img.shields.io/github/stars/PRIME-RL/Entropy-Mechanism-of-RL)
- [LLaSA-TTS-GRPO](https://github.com/channel-io/ch-tts-llasa-rl-grpo): TTS fine-tuning with GRPO optimization based on LLASA models ![GitHub Repo stars](https://img.shields.io/github/stars/channel-io/ch-tts-llasa-rl-grpo)
- [PF-PPO](https://arxiv.org/abs/2409.06957): Policy Filtration for PPO based on the reliability of reward signals for more efficient and robust RLHF.
- [RACRO](https://github.com/gyhdog99/RACRO2): Build multi-modal reasoning models via decoupling it into query-conditioned captioning and text-only reasoning ![GitHub Repo stars](https://img.shields.io/github/stars/gyhdog99/RACRO2)
- [Agent Lightning](https://github.com/microsoft/agent-lightning): A flexible and extensible framework that enables seamless agent optimization for any existing agent framework. ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/agent-lightning)
- [VTool-R1](https://github.com/VTOOL-R1/vtool-r1): VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use. ![GitHub Repo stars](https://img.shields.io/github/stars/VTOOL-R1/vtool-r1)
- [Kimina-Prover-RL](https://github.com/project-numina/kimina-prover-rl/tree/main/recipe/kimina_prover_rl): Training pipeline for formal theorem proving, based on a paradigm inspired by DeepSeek-R1.
- [RL-PLUS](https://github.com/YihongDong/RL-PLUS): Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization.
- [rStar2-Agent](https://github.com/microsoft/rStar): Using reinforcement learning with multi-step tool-calling for math tasks, rStar2-Agent-14B reaches frontier-level math reasoning in just 510 RL training steps ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/rStar)
- [Vision-SR1](https://github.com/zli12321/Vision-SR1): Self-Rewarding Vision-Language Model via Reasoning Decomposition ![GitHub Repo stars](https://img.shields.io/github/stars/zli12321/Vision-SR1)
- [SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL): SimpleVLA-RL: A Simple yet Effective Vision-Language Action Model for Reinforcement Learning ![GitHub Repo stars](https://img.shields.io/github/stars/PRIME-RL/SimpleVLA-RL)
- [Table-R1](https://github.com/Table-R1/Table-R1): Table-R1: Inference-Time Scaling for Table Reasoning ![GitHub Repo stars](https://img.shields.io/github/stars/Table-R1/Table-R1)
- [Revisual-R1](https://github.com/CSfufu/Revisual-R1): Revisual-R1: Advancing Multimodal Reasoning From Optimized Cold Start to Staged Reinforcement Learning ![GitHub Repo stars](https://img.shields.io/github/stars/CSfufu/Revisual-R1)
- [ARES](https://github.com/shawn0728/ARES): ARES: Multimodal Adaptive Reasoning via Difficulty-Aware Token-Level Entropy Shaping ![GitHub Repo stars](https://img.shields.io/github/stars/shawn0728/ARES)
- [Meta-Bandit-LLM](https://github.com/sanxing-chen/meta-bandit-llm): Meta-Bandit-LLM: Long-horizon multiturn interactive training for meta-bandit agents ![GitHub Repo stars](https://img.shields.io/github/stars/sanxing-chen/meta-bandit-llm)

and many more awesome work listed in [recipe](recipe/README.md).
