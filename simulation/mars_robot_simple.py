# IAN-Mars-Colony Simulation
# Simple Mars Robot Simulation - Phase 1
# Author: IAN Mars Colony Project
# Date: 2026

import time
import random

class MarsRobot:
    def __init__(self, name="Optimus-01"):
        self.name = name
        self.position = [0, 0]          # x, y coordinates on Mars surface
        self.battery = 100.0            # Battery percentage
        self.dust_level = 0             # Dust accumulation (0-100)
        self.collected_regolith = 0     # Resources collected

    def move(self, direction):
        """Move the robot in a direction"""
        directions = {
            "forward": (0, 1),
            "backward": (0, -1),
            "left": (-1, 0),
            "right": (1, 0)
        }
        
        if direction in directions:
            dx, dy = directions[direction]
            self.position[0] += dx
            self.position[1] += dy
            self.battery -= random.uniform(0.5, 2.0)
            self.dust_level += random.uniform(0.1, 0.8)
            print(f"{self.name} moved {direction}. New position: {self.position}")
        else:
            print("Invalid direction! Use: forward, backward, left, right")

    def collect_regolith(self):
        """Collect Martian soil (regolith)"""
        amount = random.uniform(0.5, 3.0)
        self.collected_regolith += amount
        self.battery -= 1.5
        print(f"{self.name} collected {amount:.2f} kg of regolith. Total: {self.collected_regolith:.2f} kg")

    def status(self):
        """Show robot status"""
        print("\n=== Robot Status ===")
        print(f"Name: {self.name}")
        print(f"Position: {self.position}")
        print(f"Battery: {self.battery:.1f}%")
        print(f"Dust level: {self.dust_level:.1f}%")
        print(f"Collected regolith: {self.collected_regolith:.2f} kg")
        print("===================\n")

# ==================== Main Simulation ====================
if __name__ == "__main__":
    print("🚀 IAN Mars Colony - Simple Robot Simulation Started")
    print("==================================================\n")
    
    robot = MarsRobot("Optimus-IAN-01")
    
    # Run a simple simulation loop
    for step in range(8):
        print(f"\n--- Step {step + 1} ---")
        robot.move(random.choice(["forward", "right", "left", "backward"]))
        if random.random() > 0.6:  # 40% chance to collect resources
            robot.collect_regolith()
        robot.status()
        time.sleep(0.8)  # Small delay for realism
    
    print("✅ Simulation completed. First step toward Mars colony achieved!")
