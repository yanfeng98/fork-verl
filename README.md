## verl: Volcano Engine Reinforcement Learning for LLMs

verl is a flexible, efficient and production-ready RL training library for large language models (LLMs).

verl is flexible and easy to use with:

- **Easy extension of diverse RL algorithms**

- **Seamless integration of existing LLM infra with modular APIs**.

- **Flexible device mapping**.

- **Ready integration with popular HuggingFace models**

verl is fast with:

- **State-of-the-art throughput**.

- **Efficient actor model resharding with 3D-HybridEngine**.

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

## Prepare the dataset

```bash
python examples/data_preprocess/gsm8k.py --local_dir ./data/gsm8k --max_samples 36 --val_size 0.1
```

## Code formatting

```bash
pip install --upgrade yapf
```

Then, make sure you are at top level of verl repo and run

```bash
bash scripts/format.sh
```
