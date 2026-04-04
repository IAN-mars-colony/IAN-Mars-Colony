from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time

# ====================== Project Setup ======================
app = Ursina()
window.title = 'IAN Mars Colony Project - 3D Robotic Swarm Simulation'
window.borderless = False
window.fullscreen = False
window.exit_button.visible = False
window.fps_counter.enabled = True

# Mars Environment Colors
MARS_GROUND_COLOR = color.rgb(180, 60, 30)
SKY_COLOR = color.rgb(20, 20, 40)

# ====================== Mars 3D Environment ======================
# Ground Plane
ground = Entity(
    model='plane',
    scale=(120, 1, 120),
    texture='white_cube',
    color=MARS_GROUND_COLOR,
    texture_scale=(12, 12),
    collider='mesh'
)

# Add Rocks for Realism
for _ in range(35):
    rock = Entity(
        model='cube',
        scale=(random.uniform(1.5, 5), random.uniform(0.8, 2.5), random.uniform(1.5, 4)),
        position=(random.uniform(-50, 50), 1, random.uniform(-50, 50)),
        rotation=(random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360)),
        color=color.dark_gray,
        texture='white_cube',
        collider='mesh'
    )

# Solar Charging Stations
solar_stations = []
for _ in range(6):
    station = Entity(
        model='cube',
        scale=(4, 0.3, 4),
        position=(random.uniform(-35, 35), 0.5, random.uniform(-35, 35)),
        color=color.yellow.tint(-0.3),
        collider='box'
    )
    panel = Entity(
        parent=station,
        model='cube',
        scale=(5, 2.5, 0.3),
        position=(0, 2, 0),
        rotation=(35, 0, 0),
        color=color.orange
    )
    solar_stations.append(station)

# Dust Particles
dust_particles = []
for _ in range(120):
    dust = Entity(
        model='sphere',
        scale=0.12,
        color=color.light_gray,
        position=(random.uniform(-60, 60), random.uniform(8, 35), random.uniform(-60, 60))
    )
    dust_particles.append(dust)

# ====================== Mars Robot Class ======================
class MarsRobot(Entity):
    def __init__(self, position=(0, 2, 0), robot_color=color.azure):
        super().__init__(
            model='cube',
            scale=(1.1, 2.4, 0.9),
            position=position,
            color=robot_color,
            collider='box'
        )
        # Head
        self.head = Entity(parent=self, model='sphere', scale=0.65, position=(0, 1.3, 0), color=color.white)
        # Arms
        self.left_arm = Entity(parent=self, model='cube', scale=(0.35, 1.6, 0.35), position=(-0.85, 0.6, 0), rotation=(0, 0, 35))
        self.right_arm = Entity(parent=self, model='cube', scale=(0.35, 1.6, 0.35), position=(0.85, 0.6, 0), rotation=(0, 0, -35))
        
        self.speed = 9.0
        self.energy = 100.0
        self.target = None

    def update(self):
        if self.target and self.energy > 10:
            direction = (self.target - self.position).normalized()
            self.position += direction * self.speed * time.dt
            self.look_at(self.target + Vec3(0, 1, 0))
            
            # Energy consumption
            self.energy -= 0.8 * time.dt
            
            if self.energy < 0:
                self.energy = 0

        # Keep robot above ground
        if self.y < 1.2:
            self.y = 1.2

# ====================== Create 10 Robots ======================
robots = []
for i in range(10):
    robot_color = color.azure if i < 6 else color.orange
    robot = MarsRobot(
        position=(random.uniform(-18, 18), 2, random.uniform(-18, 18)),
        robot_color=robot_color
    )
    robots.append(robot)

# ====================== Simple Agent (Ready for RL) ======================
class SimpleAgent:
    def __init__(self):
        self.memory = deque(maxlen=3000)
    
    def get_action(self, state):
        # Placeholder - will be replaced with LSTM + Double DQN later
        return np.random.uniform(-1.0, 1.0, 3)

agent = SimpleAgent()

# ====================== Camera and Lighting ======================
EditorCamera()
AmbientLight(color=color.rgb(90, 70, 55))
DirectionalLight(y=15, rotation=(50, -30, 0), color=color.white, intensity=1.2)

# ====================== On-Screen Information ======================
info_text = Text(
    text='IAN Mars Colony Project\n3D Robotic Swarm Simulation - Phase XIV',
    origin=(0, 0),
    y=0.42,
    scale=1.4,
    background=True,
    color=color.white
)

# ====================== Update Function ======================
def update():
    avg_energy = sum(robot.energy for robot in robots) / len(robots)
    
    info_text.text = (
        f'IAN Mars Colony Project\n'
        f'Robots Active: {len(robots)} | Average Energy: {avg_energy:.1f}%\n'
        f'Phase XIV - 3D Swarm Simulation'
    )
    
    # Update each robot
    for robot in robots:
        if robot.energy > 25 and (not robot.target or random.random() < 0.015):
            # Choose new target (random or near solar station)
            if random.random() < 0.4:
                robot.target = Vec3(random.uniform(-45, 45), 1, random.uniform(-45, 45))
            else:
                station = random.choice(solar_stations)
                robot.target = station.position + Vec3(0, 1, 0)
        
        # Recharge near solar stations
        for station in solar_stations:
            if distance(robot.position, station.position) < 6.0:
                robot.energy = min(100.0, robot.energy + 22 * time.dt)

    # Update dust particles
    for dust in dust_particles:
        dust.y -= 0.08
        if dust.y < 0.5:
            dust.y = random.uniform(25, 45)
            dust.x = random.uniform(-60, 60)
            dust.z = random.uniform(-60, 60)

# ====================== Input Handling ======================
def input(key):
    if key == 'escape':
        application.quit()
    if key == 'r' or key == 'R':
        for robot in robots:
            robot.energy = 100.0
        print("All robots recharged!")

# ====================== Start Simulation ======================
print("🚀 IAN Mars Colony Project - 3D Robotic Swarm Simulation Started")
print("Controls:")
print("- Right Click + Drag: Rotate Camera")
print("- Scroll: Zoom")
print("- Press R: Recharge all robots")

app.run()
