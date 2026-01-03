import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import time

from greenhouse_backend import GreenhouseEnvironment


class FuzzyGreenhouseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Adaptive Fuzzy Control + RL Optimization")
        
        self.env = GreenhouseEnvironment(target_temp=25.0)
        self.is_running = False
        self.time_step = 0
        
        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        self.btn_start = tk.Button(control_frame, text="Start Simulation", command=self.toggle_sim, bg="green", fg="white")
        self.btn_start.pack(side=tk.LEFT, padx=5)
        
        tk.Label(control_frame, text="External Heat (Sun):").pack(side=tk.LEFT, padx=5)
        self.scale_heat = tk.Scale(control_frame, from_=0, to=5, orient=tk.HORIZONTAL, resolution=0.1)
        self.scale_heat.set(1.5)
        self.scale_heat.pack(side=tk.LEFT)

        stats_frame = tk.Frame(root)
        stats_frame.pack(side=tk.TOP, fill=tk.X, padx=10)
        
        self.lbl_temp = tk.Label(stats_frame, text="Temp: 20.0°C", font=("Arial", 14))
        self.lbl_temp.pack(side=tk.LEFT, padx=20)
        
        self.lbl_fan = tk.Label(stats_frame, text="Fan: 0%", font=("Arial", 14))
        self.lbl_fan.pack(side=tk.LEFT, padx=20)

        self.lbl_reward = tk.Label(stats_frame, text="RL Reward: 0", font=("Arial", 12), fg="blue")
        self.lbl_reward.pack(side=tk.LEFT, padx=20)

        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_ylim(15, 35)
        self.line_temp, = self.ax.plot([], [], 'r-', label='Temp')
        self.line_target, = self.ax.plot([], [], 'g--', label='Target')
        self.ax.legend()
        self.ax.set_title("Real-Time Temperature Control")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.x_data = []
        self.y_temp = []
        self.y_target = []

    def toggle_sim(self):
        if self.is_running:
            self.is_running = False
            self.btn_start.config(text="Resume")
        else:
            self.is_running = True
            self.btn_start.config(text="Pause")
            self.run_step()

    def run_step(self):
        if not self.is_running:
            return

        external_heat = self.scale_heat.get()

        curr_temp, fan_power, reward = self.env.step(external_heat)
        self.time_step += 1

        self.lbl_temp.config(text=f"Temp: {curr_temp:.1f}°C")
        self.lbl_fan.config(text=f"Fan: {int(fan_power)}% (RL Modified)")
        self.lbl_reward.config(text=f"RL Reward: {reward:.2f}")

        self.x_data.append(self.time_step)
        self.y_temp.append(curr_temp)
        self.y_target.append(self.env.target_temp)
        
        if len(self.x_data) > 50:
            self.x_data.pop(0)
            self.y_temp.pop(0)
            self.y_target.pop(0)

        self.line_temp.set_data(self.x_data, self.y_temp)
        self.line_target.set_data(self.x_data, self.y_target)
        self.ax.set_xlim(min(self.x_data), max(self.x_data)+1)
        
        self.canvas.draw()

        self.root.after(100, self.run_step)

if __name__ == "__main__":
    root = tk.Tk()
    gui = FuzzyGreenhouseGUI(root)
    root.mainloop()