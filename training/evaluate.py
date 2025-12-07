# training/evaluate.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

from env.mini_chess_env import MiniChessEnv
from agent.dqn import DQN


def evaluate(num_episodes: int = 100, model_path: str = "mini_chess_dqn.pt", device: str = "cpu"):
    device = torch.device(device)
    env = MiniChessEnv()
    model = DQN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_reward = 0.0

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            legal_actions = env.legal_actions()
            if not legal_actions:
                break

            state_t = torch.from_numpy(state).unsqueeze(0).to(device)

            with torch.no_grad():
                q_values = model(state_t)[0].cpu().numpy()

            mask = np.full_like(q_values, -1e9, dtype=np.float32)
            mask[legal_actions] = 0.0
            q_masked = q_values + mask
            action = int(np.argmax(q_masked))

            next_state, reward, done, _ = env.step(action)
            state = next_state
            ep_reward += reward

        total_reward += ep_reward

    avg_reward = total_reward / num_episodes
    print(f"Average reward over {num_episodes} evaluation episodes: {avg_reward:.3f}")


if __name__ == "__main__":
    evaluate()
