import random
import torch
import torch.nn as nn
import torch.optim as optim
import pygame
import numpy as np
from collections import deque

ACTIONS = ["explore", "collect", "charge", "return", "idle"]
GRID_SIZE = 20
CELL_SIZE = 30

# ================= LSTM + DOUBLE DQN ================= #
class LSTMDoubleDQN(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, output_size=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = lstm_out[:, -1, :]
        return self.fc(out)


# ================= SWARM BRAIN ================= #
class SwarmBrain:
    def __init__(self):
        self.model = LSTMDoubleDQN()
        self.target = LSTMDoubleDQN()
        self.memory = deque(maxlen=30000)

        self.gamma = 0.97
        self.epsilon = 1.0
        self.epsilon_decay = 0.993
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.tau = 0.005

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00035)
        self.loss_fn = nn.SmoothL1Loss()

        self.update_target()

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def soft_update(self):
        for target_param, param in zip(self.target.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def act(self, state_seq):
        if random.random() < self.epsilon:
            return random.randint(0, len(ACTIONS)-1)

        state_tensor = torch.FloatTensor(state_seq).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state_seq, action, reward, next_seq):
        self.memory.append((state_seq, action, reward, next_seq))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)

        q_values = self.model(states).gather(1, actions).squeeze()

        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions).squeeze()
            target_q = rewards + self.gamma * next_q

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.soft_update()


# ================= ENVIRONMENT ================= #
class MarsEnvironment:
    def __init__(self):
        self.solar = 600.0
        self.critical_storm = False

    def update(self):
        if random.random() < 0.085:
            self.solar = max(35, self.solar - 295)
            if random.random() < 0.20:
                self.critical_storm = True
        else:
            self.solar = min(810, self.solar + 21)
            self.critical_storm = False


class MarsMap:
    def __init__(self):
        self.resources = {}
        self.generate()

    def generate(self):
        self.resources.clear()
        for _ in range(100):
            pos = (random.randint(0, 19), random.randint(0, 19))
            self.resources[pos] = random.uniform(8.0, 42.0)

    def extract(self, pos):
        if pos in self.resources and self.resources[pos] > 0:
            amount = min(self.resources[pos], random.uniform(1.8, 5.8))
            self.resources[pos] -= amount
            if self.resources[pos] <= 0.4:
                del self.resources[pos]
            return amount
        return 0.0


class EnergyGrid:
    def __init__(self):
        self.stations = [(4,4), (4,16), (16,4), (16,16)]

    def nearest(self, x, y):
        return min(self.stations, key=lambda s: abs(s[0]-x) + abs(s[1]-y))


# ================= ROBOT ================= #
class Robot:
    broadcast_memory = {}
    _lock = type('Lock', (), {'__enter__': lambda s: None, '__exit__': lambda s,*a: None})()

    def __init__(self, name, brain, base):
        self.name = name
        self.brain = brain
        self.base = base
        self.x = base[0]
        self.y = base[1]
        self.battery = 100.0
        self.carrying = 0.0
        self.visited = set()
        self.last_positions = deque(maxlen=10)
        self.history = deque(maxlen=8)
        self.colony = None

    def get_state(self, colony):
        station = colony.grid.nearest(self.x, self.y)
        dist = abs(self.x - station[0]) + abs(self.y - station[1])

        state = [
            self.battery / 100.0,
            self.carrying / 14.0,
            1.0 if (self.x, self.y) == self.base else 0.0,
            1.0 if (self.x, self.y) in colony.map.resources else 0.0,
            dist / 32.0,
            len(Robot.broadcast_memory) / 45.0,
            self.x / 20.0,
            self.y / 20.0,
            colony.env.solar / 820.0,
            1.0 if colony.env.critical_storm else 0.0,
            len(colony.map.resources) / 110.0,
            0.5
        ]

        self.history.append(state)
        while len(self.history) < 8:
            self.history.append(state)

        return list(self.history)

    def smart_explore(self):
        directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        best = None
        best_score = -9999.0

        for dx, dy in directions:
            nx = max(0, min(19, self.x + dx))
            ny = max(0, min(19, self.y + dy))
            if (nx, ny) in self.last_positions[-5:]: continue

            score = 8 if (nx, ny) not in self.visited else -5
            if (nx, ny) in Robot.broadcast_memory: score += 13
            if (nx, ny) in self.colony.map.resources: score += 22

            if score > best_score:
                best_score = score
                best = (nx, ny)

        if best:
            self.x, self.y = best
        else:
            self.x += random.choice([-1, 0, 1])
            self.y += random.choice([-1, 0, 1])
            self.x = max(0, min(19, self.x))
            self.y = max(0, min(19, self.y))

    def move_towards(self, target):
        tx, ty = target
        if self.x < tx: self.x += 1
        elif self.x > tx: self.x -= 1
        if self.y < ty: self.y += 1
        elif self.y > ty: self.y -= 1


# ================= COLONY ================= #
class Colony:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((GRID_SIZE * CELL_SIZE + 320, GRID_SIZE * CELL_SIZE))
        pygame.display.set_caption("IAN Mars Colony - Phase XIII Fixed")
        self.font = pygame.font.SysFont("consolas", 18)
        self.clock = pygame.time.Clock()

        self.env = MarsEnvironment()
        self.map = MarsMap()
        self.grid = EnergyGrid()
        self.base = (10, 10)
        self.resources_collected = 0.0

        self.brain = SwarmBrain()
        self.robots = [Robot(f"Optimus-{i}", self.brain, self.base) for i in range(10)]

        for r in self.robots:
            r.colony = self

    def draw(self, episode, step):
        self.screen.fill((15, 15, 35))

        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if (x, y) in self.map.resources:
                    pygame.draw.rect(self.screen, (0, 220, 120), rect)
                pygame.draw.rect(self.screen, (50, 50, 70), rect, 1)

        for sx, sy in self.grid.stations:
            pygame.draw.circle(self.screen, (255, 240, 0), 
                             (sx * CELL_SIZE + CELL_SIZE//2, sy * CELL_SIZE + CELL_SIZE//2), CELL_SIZE//2 - 5)

        pygame.draw.rect(self.screen, (30, 120, 255), 
                        (self.base[0]*CELL_SIZE, self.base[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE), 5)

        for robot in self.robots:
            if robot.battery <= 0: continue
            color = (255, 80, 80) if robot.battery > 40 else (255, 30, 30)
            pygame.draw.circle(self.screen, color, 
                             (robot.x*CELL_SIZE + CELL_SIZE//2, robot.y*CELL_SIZE + CELL_SIZE//2), CELL_SIZE//2 - 7)

        if self.env.critical_storm:
            overlay = pygame.Surface((GRID_SIZE*CELL_SIZE, GRID_SIZE*CELL_SIZE))
            overlay.set_alpha(90)
            overlay.fill((220, 200, 80))
            self.screen.blit(overlay, (0, 0))

        info = [
            f"Episode: {episode}   Step: {step}",
            f"Resources: {self.resources_collected:.1f}",
            f"Epsilon: {self.brain.epsilon:.3f}",
            f"Solar: {self.env.solar:.0f} {'☢️' if self.env.critical_storm else ''}",
            f"Active Robots: {sum(1 for r in self.robots if r.battery > 0)}/10"
        ]
        for i, line in enumerate(info):
            text = self.font.render(line, True, (255,255,255))
            self.screen.blit(text, (GRID_SIZE*CELL_SIZE + 25, 30 + i*32))

        pygame.display.flip()

    def step(self):
        total_reward = 0.0
        for robot in self.robots:
            if robot.battery <= 0: continue

            state_seq = robot.get_state(self)
            action_idx = self.brain.act(state_seq)
            action = ACTIONS[action_idx]

            reward = self._execute_action(robot, action)
            next_seq = robot.get_state(self)

            self.brain.remember(state_seq, action_idx, reward, next_seq)
            total_reward += reward

            robot.battery -= 1.05
            if robot.battery <= 0:
                total_reward -= 75

        self.brain.train()
        return total_reward

    def _execute_action(self, robot, action):
        reward = -0.25

        if action == "explore":
            robot.smart_explore()
            reward -= 0.1

        elif action == "collect":
            pos = (robot.x, robot.y)
            amount = self.map.extract(pos)
            robot.carrying += amount
            reward += amount * 3.6

            with Robot._lock:
                Robot.broadcast_memory[pos] = True

        elif action == "charge":
            station = self.grid.nearest(robot.x, robot.y)
            if (robot.x, robot.y) == station:
                charge = (self.env.solar / 1000) * (7 if self.env.critical_storm else 17)
                robot.battery = min(100.0, robot.battery + charge)
                reward += charge * 1.4
            else:
                robot.move_towards(station)
                reward -= 0.2

        elif action == "return":
            if (robot.x, robot.y) == self.base:
                self.resources_collected += robot.carrying
                reward += robot.carrying * 7.0
                print(f"📦 {robot.name} delivered {robot.carrying:.2f}")
                robot.carrying = 0.0
            else:
                robot.move_towards(self.base)
                reward -= 0.25

        elif action == "idle":
            reward -= 0.15

        robot.visited.add((robot.x, robot.y))
        robot.last_positions.append((robot.x, robot.y))
        return reward

    def reset(self):
        self.map.generate()
        Robot.broadcast_memory.clear()
        self.resources_collected = 0.0
        for robot in self.robots:
            robot.battery = 100.0
            robot.carrying = 0.0
            robot.visited.clear()
            robot.last_positions.clear()
            robot.history.clear()
            robot.x, robot.y = self.base


# ================= MAIN ================= #
if __name__ == "__main__":
    print("🚀 IAN Mars Colony - Phase XIII Fixed Version\n")
    random.seed(42)

    colony = Colony()

    for episode in range(1, 121):
        print(f"\n=== EPISODE {episode} ===")
        for step in range(75):
            colony.step()
            colony.env.update()
            colony.draw(episode, step)
            colony.clock.tick(15)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

        print(f"🏗️ Resources: {colony.resources_collected:.2f} | Epsilon: {colony.brain.epsilon:.3f}")

        if colony.resources_collected > 520:
            print("\n🏆 MISSION SUCCESS!")
            break

        colony.reset()

    pygame.quit()
    print("\n✅ Simulation Complete")
