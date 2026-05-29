# InteractCS-RL

**Reinforcing Real-world Service Agents: Balancing Utility and Cost in Task-Oriented Dialogue**

*ICML 2026*

[Ning Gao]()\*, [Wei Zhang]()\*, Yuqin Dai, Ling Shi, Ziyin Wang, Yujie Wang, Wei He, [Jinpeng Wang]()\†, [Chaozheng Wang]()\B

[\* Equal contribution, † Team leader, B Corresponding author]

[[Paper]](https://arxiv.org/abs/2602.22697) [[Data]](data/rl_training_data_paper)

## Overview

InteractCS-RL reframes task-oriented dialogue (TOD) as a **multi-granularity reinforcement learning** process. Instead of imitating static expert trajectories, our framework trains service agents through closed-loop interaction with persona-driven user simulators, enabling them to **balance empathetic communication with budget-aware decision-making**.

<p align="center">
  <img src="assets/framework.png" width="90%" alt="InteractCS-RL Framework"/>
</p>

### Key Components

- **User-Centric Interaction Framework**: A dynamic training gym driven by realistic persona profiles (intrinsic traits + extrinsic demands) that generates diverse, non-cooperative user interactions.
- **Cost-aware Multi-turn Policy Optimization (CMPO)**: A hybrid advantage estimation strategy combining:
  - *Session-level Outcome Utility* — user satisfaction at dialogue end
  - *Process Credit Assignment* — turn-level quality via a Generative Reward Model (GenRM)
  - *Cost-aware Lagrange Penalty* — PID-controlled dynamic cost signals for operational budget constraints (e.g., voucher rate)

### Main Results

On the **FoodDeliveryService (FDS)** benchmark, InteractCS-RL (Qwen-2.5-7B) outperforms GPT-4.1, DeepSeek-v3.2, and other baselines across all three evaluation dimensions (Satisfaction, Communication Quality, Logic Quality), while maintaining voucher cost below the 30% constraint.

## Repository Structure

```
InteractCS-RL/
├── framework/
│   ├── verl-customer-stable/          # RL training framework (customized verl)
│   │   ├── customer_service_related/  # Customer-service-specific extensions
│   │   │   ├── interactions/          # Multi-turn RL environment
│   │   │   ├── reward_function/       # Hybrid reward (GenRM + rule-based)
│   │   │   ├── data_preprocess/       # Data format conversion
│   │   │   ├── config/               # Training YAML configs
│   │   │   └── sh/                   # Training launch scripts
│   │   └── verl/                     # Core verl library
│   └── customer_service_simulation_framework/  # Dialogue simulation & evaluation
│       ├── agents/                   # Agent implementations
│       ├── session/                  # Session types (RL simulation, evaluation)
│       ├── llm/                      # LLM client backends (OpenAI, vLLM)
│       ├── config/                   # Simulation YAML configs
│       └── core/                     # Registry & type system
├── code/
│   ├── customer_service_evaluation/  # Standalone multi-dimensional evaluation
│   │   ├── evaluators/              # Speech, logic, compensation, format evaluators
│   │   ├── simulators/              # User & customer service simulators
│   │   └── core/                    # Parallel evaluation manager
│   └── prompt/                      # Prompt templates for simulation & evaluation
├── data/
│   └── rl_training_data_paper/      # FDS benchmark dataset
│       ├── 8000_sample_ordered_no_transfer/          # Scene 1 (Hard)
│       │   ├── train.parquet        # 7,680 samples
│       │   └── test.parquet         # 256 samples
│       └── 8000_sample_ordered_no_transfer_eazylevel/ # Scene 2 (Easy)
│           ├── train.parquet        # 7,424 samples
│           └── test.parquet         # 256 samples
└── prompt_for_verify/               # GenRM reward & evaluation prompt templates
```

## Setup

### Requirements

- Python >= 3.9
- PyTorch >= 2.1
- 8x NVIDIA A100 GPUs (for RL training)
- 2x NVIDIA H20 GPUs (for user simulator / reward model inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/NingGaoAi/InteractCS-RL.git
cd InteractCS-RL

# Install verl and dependencies
cd framework/verl-customer-stable
pip install -e .
cd ../..

# Install additional dependencies
pip install vllm pyarrow
```

### Environment Variables

The hybrid reward function supports two modes for the Generative Reward Model (GenRM):

**Option A: Self-hosted vLLM server**

```bash
export REWARD_API_URL="http://localhost:8000/v1/chat/completions"
export REWARD_API_KEY="none"                  # Not required for local vLLM
export REWARD_MODEL="<your_local_model>"      # Model name served by vLLM
export REWARD_MAX_WORKERS=32                  # Parallel workers
```

**Option B: External API provider**

```bash
export REWARD_API_URL="<your_api_url>"        # External LLM API endpoint
export REWARD_API_KEY="<your_api_key>"        # API key
export REWARD_MODEL="gpt-4.1"                # Model for reward scoring
export REWARD_MAX_WORKERS=32                  # Parallel workers
```

## Usage

### 1. RL Training (CMPO)

Configure the training YAML in `framework/verl-customer-stable/customer_service_related/config/`, then launch:

```bash
cd framework/verl-customer-stable/customer_service_related
bash sh/rl_hybrid_reward/qwen25_7b_sft_1200_data_grpo_8000_data_hybrid_reward.sh
```

Key configuration options in the YAML config:

| Parameter | Description |
|-----------|-------------|
| `actor_rollout_ref.model.path` | Path to the SFT base model |
| `actor_rollout_ref.rollout.n` | Number of rollouts per prompt |
| `custom.interaction.name` | Interaction type (`user_simulation`) |
| `custom.interaction.max_turns` | Maximum dialogue turns (default: 15) |
| `custom.interaction.voucher_target_rate` | Cost constraint target (e.g., 0.3) |

### 2. Dialogue Simulation

Run multi-turn dialogue simulation with the User-Centric Interaction Framework:

```bash
cd framework/customer_service_simulation_framework
python main.py --config config/<your_config>.yaml
```

Override config parameters via CLI:

```bash
python main.py --config config/default.yaml \
    --agent.customer.llm.model=<model_name> \
    --session.RLSimulation.num_conversations=100
```

### 3. Evaluation

Run multi-dimensional evaluation (communication, logic, compensation, format):

```bash
cd code/customer_service_evaluation
python main.py --config config/config_yaml/config.yaml
```

Or run evaluation-only mode:

```bash
python main_evaluation.py --config config/config_yaml/config_evaluation.yaml
```

## Citation

```bibtex
@inproceedings{gao2026interactcsrl,
  title={Reinforcing Real-world Service Agents: Balancing Utility and Cost in Task-Oriented Dialogue},
  author={Gao, Ning and Zhang, Wei and Dai, Yuqin and Shi, Ling and Wang, Ziyin and Wang, Yujie and He, Wei and Wang, Jinpeng and Wang, Chaozheng},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## License

This project is released under the [Apache 2.0 License](LICENSE).

## Acknowledgements

This project builds upon [verl](https://github.com/volcengine/verl) for the RL training infrastructure.
