# 🚀 IAN Mars Colony

**Open-source AI Swarm Framework for the First Self-Sustaining Robot Colony on Mars**

We are building an intelligent, autonomous AI system designed for Tesla Optimus and future Mars robots to explore, collect resources, manage energy, and build a self-sustaining colony on the Red Planet.

![Mars Simulation](https://via.placeholder.com/800x400/1a1a2e/0f0?text=Mars+Colony+Simulation)

---

### ✨ Current Status — Phase XIII Completed

- Fully functional simulation with 10 autonomous robots
- LSTM + Double DQN with soft target updates and experience replay
- Realistic Mars environment (solar power, dynamic dust storms, charging stations)
- Smart exploration, resource collection, and swarm intelligence via broadcast memory
- Real-time Pygame visualization with live dashboard

---

### 🎯 Project Goals

- Develop advanced AI for Martian navigation, resource management, and energy optimization
- Create a realistic physics-based simulation of a Mars colony
- Enable true **swarm intelligence** between hundreds of robots
- Prepare the framework for real Optimus robots and future Mars missions
- Build an open, collaborative, and community-driven project

---

### 🛠 Current Technologies

- **Python 3.10+**
- **PyTorch** — LSTM + Double DQN (Deep Reinforcement Learning)
- **Pygame** — Real-time visualization
- **NumPy**
- Planned: ROS2, Unity 3D Simulation, Multi-agent RL

---

### 🚀 Quick Start

```bash
git clone https://github.com/IAN-mars-colony/IAN-Mars-Colony.git
cd IAN-Mars-Colony

# Install dependencies
pip install torch torchvision torchaudio pygame numpy

# Run the simulation
cd simulation
python mars_robot_simple.py
