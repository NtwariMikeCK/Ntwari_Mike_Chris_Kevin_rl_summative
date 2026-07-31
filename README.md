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

## Environment design (updated)
- Drone starts a short hop from the hospital, at a full 100% battery
- Charging station is fixed at the midpoint of the hospital-clinic route — always reachable
  within roughly half the total mission distance, on either leg of the trip
- Recharging fills the battery to 100% (was a flat +25%)
- Battery drain is halved (movement 2%→1%, hover 1%→0.5%, wait 0.5%→0.25%, storm surcharge
  2%→1%), giving the agent roughly double the moves per charge to explore before it needs to
  commit to a known objective
- Fixed a minor distance-shaping edge case where the reward could compare distance-to-old-objective
  against distance-to-new-objective on the rare step where crossing the low-battery threshold
  flips the current objective mid-step

These changes took the naive, hazard-blind heuristic's success rate from ~22% to ~70% (and it
solved 9/10 rollouts in the recorded demo video), and eliminated battery depletion as a common
failure mode entirely — confirming the mission is comfortably feasible; if any algorithm still
returns a 0% success rate, that now points to that algorithm's own exploration/training
dynamics rather than the environment.

## Training notebooks (updated design)
Each of the four training notebooks (`01_dqn_training.ipynb`, `02_reinforce_training.ipynb`,
`03_ppo_training.ipynb`, `04_a2c_training.ipynb`) now trains a **single configuration** — the
best hyperparameters found during the original 10-run sweep — for **500,000 timesteps**,
instead of re-running the full 10-configuration sweep:
- Training is **checkpointed every 50,000 steps**. If Colab disconnects partway through, just
  re-run the training cell — it resumes from the last checkpoint instead of retraining from
  step 0.
- Evaluation now reports a **mission milestone funnel** (% of episodes that reached the
  hospital / picked up the package / visited the charger / reached the clinic / delivered) in
  addition to the summary metrics, so you can see exactly where a policy is failing.
- The rollout video now concatenates **10 episodes back-to-back** (~15-20 seconds), since a
  single ~20-30 step episode at 30fps was too short to actually watch.
- Each notebook exposes `continue_training(extra_timesteps)`: once you've compared all four and
  picked a winner, use this to keep training that one further without starting over.

`05_comparison_and_analysis.ipynb` also now includes a milestone-funnel comparison chart across
all four algorithms, in addition to the reward-curve, convergence, generalization, and stability
comparisons.


## Local video recording (uv)
```bash
uv sync
uv run main.py --episodes 3 --policy heuristic          # environment demo, no trained model needed
uv run training/play_best_agent.py --algo ppo --model models/ppo/ppo_best_1.zip
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
