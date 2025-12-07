# MiniChess-RL: Deep Q-Network Agent for Simplified Chess

This project implements a 4×4 mini-chess environment and a Deep Q-Network (DQN) agent that learns to play through reinforcement learning.

---

## Features

- Custom 4×4 mini-chess environment (King + Rook vs King)
- State encoded as a 4×4×4 tensor (piece channels + turn)
- Discrete action space with 256 possible moves
- DQN agent implemented in PyTorch
- Replay buffer + target network
- Training and evaluation scripts

---

## Project Structure

```text
env/
  __init__.py
  mini_chess_env.py

agent/
  dqn.py
  replay_buffer.py

training/
  train.py
  evaluate.py

requirements.txt
README.md
