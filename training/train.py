# training/train.py

import numpy as np
import torch
import torch.nn.functional as F

from env.mini_chess_env import MiniChessEnv
from agent.dqn import DQN, create_optimizer
from agent.replay_buffer import ReplayBuffer


def select_action(model, state, legal_actions, epsilon, device):
    if np.random.rand() < epsilon:
        return int(np.random.choice(legal_actions))
    else:
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)  # (1, H, W, C)
        with torch.no_grad():
            q_values = model(state_tensor)[0]  # (num_actions,)
        q_values = q_values.cpu().numpy()

        # mask illegal actions
        mask = np.full_like(q_values, -1e9, dtype=np.float32)
        mask[legal_actions] = 0.0
        q_values_masked = q_values + mask
        return int(np.argmax(q_values_masked))


def train(
    num_episodes: int = 2000,
    buffer_capacity: int = 50000,
    batch_size: int = 64,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    target_update_interval: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
):
    device = torch.device(device)
    env = MiniChessEnv()
    model = DQN().to(device)
    target_model = DQN().to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = create_optimizer(model, lr)
    replay_buffer = ReplayBuffer(buffer_capacity)

    epsilon = epsilon_start

    all_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            legal_actions = env.legal_actions()
            if not legal_actions:
                # no moves → end episode
                break

            action = select_action(model, state, legal_actions, epsilon, device)
            next_state, reward, done, _ = env.step(action)

            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                states_t = torch.from_numpy(states).to(device)   # (B, H, W, C)
                next_states_t = torch.from_numpy(next_states).to(device)
                actions_t = torch.from_numpy(actions).to(device)
                rewards_t = torch.from_numpy(rewards).to(device)
                dones_t = torch.from_numpy(dones).to(device)

                # Q(s,a)
                q_values = model(states_t)  # (B, num_actions)
                q_sa = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (B,)

                # max_a' Q_target(s', a')
                with torch.no_grad():
                    q_next = target_model(next_states_t)  # (B, num_actions)
                    max_q_next, _ = q_next.max(dim=1)     # (B,)
                    targets = rewards_t + gamma * max_q_next * (1.0 - dones_t)

                loss = F.mse_loss(q_sa, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # epsilon decay
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        all_rewards.append(episode_reward)

        # update target network
        if (episode + 1) % target_update_interval == 0:
            target_model.load_state_dict(model.state_dict())

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(all_rewards[-50:])
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"eps={epsilon:.3f} | avg_reward(last50)={avg_reward:.3f}")

    # ذخیره مدل
    torch.save(model.state_dict(), "mini_chess_dqn.pt")
    return all_rewards


if __name__ == "__main__":
    train()
