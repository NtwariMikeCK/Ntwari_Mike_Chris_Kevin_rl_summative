# Autonomous Medical Drone Delivery — RL Summative

Custom Gymnasium environment simulating a medical supply drone that picks up
packages at a hospital and delivers them to a rural clinic, while managing
battery, avoiding storms/mountains/no-fly zones, and reacting to a
randomized mission each episode.

## Project layout
```
project_root/
├── pyproject.toml
├── main.py                  # demo runner (cinematic camera + heuristic/random agent)
├── environment/
│   ├── custom_env.py        # Gymnasium env: spaces, dynamics, reward function
│   └── rendering.py         # Pygame renderer: terrain, sprites, camera system
├── notebooks/                # Google Colab training notebooks (GPU + Google Drive)
│   ├── 01_dqn_training.ipynb
│   ├── 02_reinforce_training.ipynb   # custom PyTorch REINFORCE (not in stable-baselines3)
│   ├── 03_ppo_training.ipynb
│   ├── 04_a2c_training.ipynb
│   └── 05_comparison_and_analysis.ipynb   # cross-algorithm plots & tables, run last
├── training/
│   └── play_best_agent.py   # local playback with live GUI + verbose terminal, for the video demo
├── models/, logs/, assets/, tests/
```

## Training (Google Colab)
Open each notebook in `notebooks/` in Colab (Runtime → Change runtime type → **GPU**), run all cells top
to bottom. Each notebook:
- Mounts your Google Drive and creates `MyDrive/student_name_rl_summative/{models,logs,results,assets}/<algo>/`
- Writes a synced copy of `environment/` to Drive so all four notebooks share the exact same environment
- Defines and documents **10 hyperparameter configurations** with the reasoning behind each choice
- Trains on GPU, logs to TensorBoard, and evaluates every run on a fixed set of 30 unseen seeds
- Saves a hyperparameter-results CSV, reward-curve plots, an algorithm-specific plot (DQN loss curve /
  policy-entropy curve for the others), the best model, its full Section-15 metrics, a couple of
  "hard scenario" generalization tests, and a rollout video — all to Drive
- Run `05_comparison_and_analysis.ipynb` last, after all four have completed at least once, to get the
  combined cross-algorithm comparison plots and the final results table for the report

**Before you rely on the numbers for the report:** the default `TOTAL_TIMESTEPS_PER_RUN = 100_000` per
config (×10 configs ×4 algorithms) is tuned to fit comfortably in a Colab GPU session. Bump it up if you
have the runtime budget — REINFORCE in particular tends to need more environment interactions than PPO/A2C.

## Local video recording (uv)
```bash
uv sync
uv run main.py --episodes 3 --policy heuristic          # environment demo, no trained model needed
uv run training/play_best_agent.py --algo ppo --model models/ppo/ppo_01.zip   # after downloading a trained model from Drive
```

## Controls (once the intro flythrough finishes)
- `M` or `F` — toggle manual camera mode
- Arrow keys / `WASD` — pan camera (manual mode)
- `+` / `-` — zoom (manual mode)
- `Esc` — quit

## Camera behavior
1. Wide shot of the entire map
2. Pan/zoom to the hospital
3. Pan/zoom to the clinic
4. Pan/zoom to the drone's start position
5. Switches to smooth follow-cam on the drone
6. Player can override with manual pan/zoom at any time, and resume follow with `M`/`F`

## Environment summary
- **Action space**: Discrete(7) — N/S/E/W, Hover, Interact (context-sensitive), Wait
- **Observation space**: drone state, mission state, hazard positions, distance features (38-dim)
- **Reward**: pickup +30, delivery +200 (+efficiency/urgency bonuses), hazard penalties
  (storm −20, no-fly −60 terminal, mountain −100 terminal), battery-aware recharge shaping,
  distance-shaping, step penalty −0.5
- **Randomized per episode**: drone start, battery, package status, hospital/clinic/charging
  positions, mountains, storms, no-fly zones (guaranteed clear of hospital/clinic/charging/drone),
  wind, mission priority

See `../assignment spec/Autonomous Medical Drone Delivery Environment Specification` for the
full design document.
