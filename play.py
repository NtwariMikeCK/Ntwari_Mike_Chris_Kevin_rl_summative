"""
Local playback of a trained agent for the video demonstration.

After training in Colab, download the relevant model file(s) from
Google Drive (student_name_rl_summative/models/<algo>/...) into this
project's models/<algo>/ folder, then run this script locally so you get
a live GUI window plus verbose terminal output — exactly what the
assignment's video requirement asks for.

Examples:
    uv run training/play_best_agent.py --algo dqn --model models/dqn/dqn_10.zip
    uv run training/play_best_agent.py --algo ppo --model models/ppo/ppo_01.zip
    uv run training/play_best_agent.py --algo reinforce --model models/reinforce/reinforce_09.pt --hidden-size 128
"""

import argparse
import sys
import time

sys.path.insert(0, ".")
from environment.custom_env import MedicalDroneEnv


def load_sb3_model(algo, path):
    if algo == "dqn":
        from stable_baselines3 import DQN as Algo
    elif algo == "ppo":
        from stable_baselines3 import PPO as Algo
    elif algo == "a2c":
        from stable_baselines3 import A2C as Algo
    else:
        raise ValueError(algo)
    return Algo.load(path)


def load_reinforce_model(path, obs_dim, n_actions, hidden_size):
    import torch
    import torch.nn as nn

    class PolicyNet(nn.Module):
        def __init__(self, obs_dim, n_actions, hidden_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_size), nn.Tanh(),
                nn.Linear(hidden_size, hidden_size), nn.Tanh(),
                nn.Linear(hidden_size, n_actions),
            )

        def forward(self, x):
            return self.net(x)

    policy = PolicyNet(obs_dim, n_actions, hidden_size)
    policy.load_state_dict(torch.load(path, map_location="cpu"))
    policy.eval()
    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "reinforce", "ppo", "a2c"], required=True)
    parser.add_argument("--model", required=True, help="Path to the saved model file")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=128, help="Only needed for --algo reinforce")
    args = parser.parse_args()

    env = MedicalDroneEnv(render_mode="human")

    if args.algo == "reinforce":
        import torch
        policy = load_reinforce_model(args.model, env.observation_space.shape[0], env.action_space.n, args.hidden_size)

        def predict(obs):
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits = policy(obs_t)
                return int(torch.argmax(logits, dim=-1).item())
    else:
        model = load_sb3_model(args.algo, args.model)

        def predict(obs):
            action, _ = model.predict(obs, deterministic=True)
            return int(action)

    print(f"Loaded {args.algo.upper()} model from {args.model}")
    print("Running", args.episodes, "episode(s). Close the window or press Esc to stop early.\n")

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed)
        env.render()
        print(f"--- Episode {ep + 1} ---")
        print(f"Start: battery={info['battery']:.0f}%  package={info['has_package']}  "
              f"priority={info['priority']}  objective={info['objective']}")

        terminated = truncated = False
        ep_reward = 0.0
        while not (terminated or truncated):
            if getattr(env.renderer, "_quit_requested", False):
                env.close()
                return
            action = predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            print(f"step={info['steps']:3d}  action={action}  reward={reward:+6.1f}  "
                  f"battery={info['battery']:5.1f}%  objective={info['objective']:<10s}  event={info['event']}")
            env.render()

        print(f"--- Episode {ep + 1} finished: return={ep_reward:.1f}  "
              f"final_event='{info['event']}'  steps={info['steps']} ---\n")
        time.sleep(1.5)

    env.close()


if __name__ == "__main__":
    main()
