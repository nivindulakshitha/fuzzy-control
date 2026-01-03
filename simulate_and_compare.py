import numpy as np
import time
from greenhouse_backend import GreenhouseEnvironment, SugenoController

def run_simulation_episode(controller_type, env, sugeno_ctrl, external_heat, target_temp, steps=50):
    """
    Runs a single simulation episode.
    controller_type: 'Mamdani' or 'Sugeno'
    """
    # Reset Environment State
    env.current_temp = 20.0 # Start at ambient
    env.current_hum = 50.0
    env.target_temp = target_temp
    env.apply_adaptation('General') # Reset to base
    env.target_temp = target_temp # Re-apply target manually just in case
    
    # For Sugeno, we need to manage physics manually since env.step() uses Mamdani logic + RL
    # But to keep physics consistent, we should reuse env's physics logic if possible.
    # However, env.step() is tightly coupled. We will replicate the physics loop here for Sugeno.
    
    temp_history = []
    fan_history = []
    error_acc = 0
    energy_acc = 0
    
    current_temp = env.current_temp
    current_hum = env.current_hum
    
    start_time = time.time()
    
    for _ in range(steps):
        # 1. Get Control Output
        if controller_type == 'Mamdani':
            # env.step() does: Error Calc -> Fuzzy -> RL -> Physics
            # We want to measure pure Mamdani first (RL epsilon=0 for 'After', or we can just use env.step)
            # The prompt asks for RL optimization comparison later.
            # For the main comparison, we'll use env.step() which includes the current RL state.
            
            # Note: env.step() updates env.current_temp internally
            t, fan, _ = env.step(external_heat, 'Vegetative')
            current_temp = t
            
        elif controller_type == 'Sugeno':
            # Compute Sugeno Output
            fan = sugeno_ctrl.compute(current_temp, current_hum, 5) # 5=Vegetative
            
            # Physics (Replicated from greenhouse_backend.py)
            cooling_effect = (fan / 18.0)
            warming_effect = external_heat
            heat_loss = 0.1 * (current_temp - 20.0)
            
            current_temp += (warming_effect - cooling_effect - heat_loss)
            
            # Humidity Physics
            current_hum -= (fan / 50.0)
            current_hum += 0.5
            current_hum = max(0, min(100, current_hum))
            
        # 2. Track Metrics
        temp_history.append(current_temp)
        fan_history.append(fan)
        error_acc += abs(current_temp - target_temp)
        energy_acc += fan
        
    end_time = time.time()
    
    # Calculate Metrics
    avg_error = error_acc / steps
    
    # Response Time: Steps until error stays below 1.0 degree for the rest of the run (or last 5 steps)
    # Simplified: First step where error < 1.0
    response_time = steps # Default if never reached
    for i, t in enumerate(temp_history):
        if abs(t - target_temp) < 1.0:
            response_time = i
            break
            
    # Smoothness: Standard Deviation of Fan Output (lower is smoother)
    smoothness = np.std(fan_history)
    
    return response_time, avg_error, energy_acc, smoothness

def run_comparison():
    print("\n=== 1. CONTROLLER COMPARISON (Mamdani vs Sugeno) ===")
    print(f"{'Case':<5} | {'Type':<8} | {'RespTime':<8} | {'AvgErr':<8} | {'Energy':<8} | {'Smooth':<8}")
    print("-" * 65)
    
    env = GreenhouseEnvironment()
    sugeno = SugenoController()
    
    # Disable RL exploration for the comparison to be fair/stable
    env.epsilon = 0.0 
    
    results = {'Mamdani': {'rt': [], 'err': [], 'en': [], 'sm': []},
               'Sugeno':  {'rt': [], 'err': [], 'en': [], 'sm': []}}
    
    for i in range(5):
        # Random Scenario
        ext_heat = np.random.uniform(0.5, 4.0)
        target = np.random.choice([18, 24, 28])
        
        # Run Mamdani
        rt_m, err_m, en_m, sm_m = run_simulation_episode('Mamdani', env, sugeno, ext_heat, target)
        
        # Run Sugeno
        rt_s, err_s, en_s, sm_s = run_simulation_episode('Sugeno', env, sugeno, ext_heat, target)
        
        # Store
        results['Mamdani']['rt'].append(rt_m)
        results['Mamdani']['err'].append(err_m)
        results['Mamdani']['en'].append(en_m)
        results['Mamdani']['sm'].append(sm_m)
        results['Sugeno']['rt'].append(rt_s)
        results['Sugeno']['err'].append(err_s)
        results['Sugeno']['en'].append(en_s)
        results['Sugeno']['sm'].append(sm_s)
        
        print(f"{i+1:<5} | Mamdani  | {rt_m:<8} | {err_m:<8.2f} | {en_m:<8.0f} | {sm_m:<8.2f}")
        print(f"{'':<5} | Sugeno   | {rt_s:<8} | {err_s:<8.2f} | {en_s:<8.0f} | {sm_s:<8.2f}")
        
    # Averages
    print("-" * 65)
    print("AVERAGES:")
    for c_type in ['Mamdani', 'Sugeno']:
        avg_rt = np.mean(results[c_type]['rt'])
        avg_err = np.mean(results[c_type]['err'])
        avg_en = np.mean(results[c_type]['en'])
        avg_sm = np.mean(results[c_type]['sm'])
        print(f"{c_type:<8} | {avg_rt:<8.1f} | {avg_err:<8.2f} | {avg_en:<8.1f} | {avg_sm:<8.2f}")

def run_rl_optimization_test():
    print("\n=== 2. RL OPTIMIZATION (Before vs After) ===")
    
    env = GreenhouseEnvironment()
    
    # 1. BEFORE TRAINING
    # Reset Q-table
    env.q_table = np.zeros((20, 3))
    env.epsilon = 0.0 # No exploration, just exploit empty table (all zeros -> action 0 -> no adjustment)
    
    print("Running 'Before' Test (Untrained)...")
    # Run a standard scenario
    rt_b, err_b, en_b, sm_b = run_simulation_episode('Mamdani', env, None, external_heat=3.0, target_temp=24.0, steps=100)
    
    # 2. TRAINING
    print("Training RL Agent (50 Episodes)...")
    env.epsilon = 0.5 # High exploration
    for episode in range(50):
        env.current_temp = np.random.uniform(15, 35)
        env.target_temp = 24.0
        for _ in range(20): # Short episodes
            env.step(3.0, 'Vegetative')
            
    # 3. AFTER TRAINING
    print("Running 'After' Test (Trained)...")
    env.epsilon = 0.0 # Exploit learned policy
    rt_a, err_a, en_a, sm_a = run_simulation_episode('Mamdani', env, None, external_heat=3.0, target_temp=24.0, steps=100)
    
    print(f"\n{'Metric':<15} | {'Before':<10} | {'After':<10} | {'Improvement':<10}")
    print("-" * 55)
    print(f"{'Avg Error':<15} | {err_b:<10.2f} | {err_a:<10.2f} | {err_b - err_a:<10.2f}")
    print(f"{'Response Time':<15} | {rt_b:<10} | {rt_a:<10} | {rt_b - rt_a:<10}")
    print(f"{'Smoothness':<15} | {sm_b:<10.2f} | {sm_a:<10.2f} | {sm_b - sm_a:<10.2f}")

if __name__ == "__main__":
    # run_comparison()
    run_rl_optimization_test()
