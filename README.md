# Smart Greenhouse Adaptive Fuzzy Climate Control System

## Overview

This project implements an intelligent climate control system for a greenhouse using **Fuzzy Logic** and **Reinforcement Learning (RL)**. It features an adaptive mechanism that adjusts control parameters based on specific plant requirements (e.g., Tomato, Lettuce, Orchid) and uses Q-Learning to optimize performance over time.

## Features

-   **Dual Fuzzy Controllers**:
    -   **Mamdani Controller**: Uses `scikit-fuzzy` with 25+ rules for complex, human-like reasoning.
    -   **Sugeno Controller**: A manually implemented 0-order Takagi-Sugeno controller for high-speed benchmarking.
-   **Adaptive Control**: Dynamically shifts fuzzy membership functions to target specific temperature and humidity zones for different plant species.
-   **Reinforcement Learning**: A Q-Learning agent that fine-tunes the fan output to minimize steady-state error.
-   **Physics Simulation**: Realistic environment modeling including external heat sources (sun), cooling efficiency, and natural heat loss.
-   **Interactive GUI**: Real-time visualization of temperature, target setpoints, and control actions.

## Installation

1. **Prerequisites**: Python 3.x
2. **Install Dependencies**:
    ```bash
    pip install numpy scikit-fuzzy matplotlib
    ```
    _(Note: Tkinter is usually included with standard Python installations)_

## Usage

### 1. Run the GUI Simulation

Start the interactive control panel to visualize the system in real-time.

```bash
python main_gui.py
```

**Controls:**

-   **Start/Pause**: Toggle the simulation.
-   **External Heat**: Simulate sun intensity (0.0 - 5.0).
-   **Plant**: Select species (General, Tomato, Lettuce, Orchid) to trigger adaptation.
-   **Stage**: Adjust growth stage (Seedling, Vegetative, Flowering).

### 2. Run Performance Comparison

Execute the automated test suite to compare Mamdani vs. Sugeno controllers and verify RL optimization.

```bash
python simulate_and_compare.py
```

This script outputs a table comparing:

-   Average Response Time
-   Average Error
-   Energy Usage
-   Smoothness Score

## System Architecture

### Backend (`greenhouse_backend.py`)

-   **`GreenhouseEnvironment`**: Core class containing the Mamdani controller, RL agent, and physics engine.
-   **`SugenoController`**: Lightweight class implementing Takagi-Sugeno inference logic.
-   **`apply_adaptation()`**: Method that reconfigures fuzzy sets based on plant species.

### Frontend (`main_gui.py`)

-   Built with `tkinter` and `matplotlib`.
-   Updates the simulation loop and renders the real-time temperature graph.

## Plant Profiles

-   **Tomato**: Target 24°C, High Humidity (70%)
-   **Lettuce**: Target 18°C, Moderate Humidity (60%)
-   **Orchid**: Target 26°C, High Humidity (80%)
-   **General**: Target 24°C, Standard Humidity (50%)
