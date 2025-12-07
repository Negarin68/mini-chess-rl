MiniChess-RL: Deep Q-Network Agent for a Custom 4×4 Chess Environment
This project implements a custom mini-chess environment (4×4) and a Deep Q-Network (DQN) reinforcement learning agent trained to optimize sequential decision-making.
The environment, agent, training pipeline, and evaluation components are fully built from scratch using Python + PyTorch.
This repository demonstrates core RL research skills including environment modeling, reward shaping, neural Q-functions, training loops, illegal move masking, and performance evaluation.
1. Project Overview
MiniChess-RL is a simplified strategic environment inspired by chess.
The objective is to train a DQN agent to make optimal moves in a constrained chess-like environment:
Board size: 4×4
White pieces: King + Rook (RL agent)
Black piece: King (random opponent)
Action space: 256 discrete actions (from-square × to-square)
State encoding: 4-channel tensor (WK, WR, BK, turn)
This reduced environment allows faster experimentation and is ideal for beginners entering reinforcement learning research.
2. Features
Custom Gym-style environment
Full MDP formalization (state, action, reward, transitions)
DQN with:
Replay buffer
Target network
Epsilon-greedy exploration
Illegal action masking
Evaluation over 100 episodes
Reward curve visualization
Modular code structure suitable for extension or publication
3. Project Structure
mini_chess_rl/
│
├── env/
│   ├── __init__.py
│   └── mini_chess_env.py        # Custom mini-chess environment
│
├── agent/
│   ├── dqn.py                   # DQN model architecture
│   └── replay_buffer.py         # Experience replay buffer
│
├── training/
│   ├── train.py                 # Training loop + saving training curve
│   └── evaluate.py              # Evaluation script
│
├── results/
│   └── reward_curve.png         # Training reward curve
│
├── requirements.txt
└── README.md
4. How to Run the Project
Install
git clone https://github.com/Negarin68/mini-chess-rl.git
cd mini-chess-rl

python3 -m venv .venv
source .venv/bin/activate     # For macOS/Linux

pip install -r requirements.txt
5. Train the Agent
python3 training/train.py
During training, you’ll see logs such as:
Episode 100/2000 | eps=0.60 | avg_reward(last50)=0.07
Episode 500/2000 | eps=0.23 | avg_reward(last50)=0.21
...
After training, a reward curve will be saved automatically:
results/reward_curve.png
6. Evaluation
Run:
python3 training/evaluate.py
Example output:
Average reward over 100 evaluation episodes: -0.582
This indicates early-stage learning. With more training (e.g., 20,000+ episodes) or reward shaping, the agent improves significantly.
7. Training Curve
Below is the moving-average reward curve (window = 50 episodes):
This curve helps visualize learning performance over time.
8. Environment Details
State (4×4×4 Tensor):
Channel 0 → White King
Channel 1 → White Rook
Channel 2 → Black King
Channel 3 → Turn indicator
Reward Function:
+1.0  → win
-1.0  → loss
 0.0  → draw
-0.01 → per-step penalty (encourages shorter games)
Action Encoding (256 total):
action = from_square * 16 + to_square
9. Future Extensions (ideal for a research paper)
Better reward shaping (distance-based incentives)
Replace random opponent with heuristic AI
Extend to PPO, A2C, or Hierarchical RL
Add self-play
Increase board size or piece variety
Connect this environment to strategic RL frameworks (e.g., H-MDP, BCS)
10. Academic Value
This project demonstrates:
✔ Ability to design RL environments
✔ Ability to implement a neural agent
✔ Understanding of exploration, replay, target networks
✔ Experimentation and evaluation
✔ Reproducible research workflow
✔ Professional code structuring
Perfect for:
Master’s / PhD applications
Research samples
Portfolio demonstrations
11. Contact
For questions or collaboration:
Negar Arianfar – 2025
GitHub: https://github.com/Negarin68
