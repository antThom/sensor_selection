# sensor_selection

# 🧠 Goal-Seeking Agent Simulation with Ursina, PyBullet, Gym, and PPO

This project implements a simple 3D **goal-seeking agent** using:

- 🎮 [Ursina](https://www.ursinaengine.org/) for real-time 3D visualization
- 🔧 [PyBullet](https://pybullet.org/) for physics simulation
- 🧪 [Gym](https://www.gymlibrary.dev/) for RL-compatible environment API
- 🤖 [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for PPO training

The simulation involves a robot agent navigating toward a randomly placed goal using reinforcement learning.

---

## 📦 Features

- ✅ Real-time 3D agent movement visualization
- ✅ Physics-based dynamics via PyBullet
- ✅ Goal-seeking task with reward shaping
- ✅ PPO training in the background
- ✅ Modular code for easy extension (obstacles, sensors, multi-agent)

---

## 🚀 Quick Start

```bash
# Step 1: Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Step 2: Create and activate environment
python -m venv sim_env
sim_env\Scripts\activate  # On Windows

# Step 3: Install dependencies from PyPI
pip install -r requirements.txt
```

--

## Documentation

Create a new directory `doxygen` and run doxygen inside of the folder with the provided `Doxyfile`.

```bash
mkdir doxygen 
cd doxygen
doxygen Doxyfile
```

