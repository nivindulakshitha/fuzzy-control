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
    
    temp_history = []
    fan_history = []
    error_acc = 0
    energy_acc = 0
    
    current_temp = env.current_temp
    current_hum = env.current_hum
    
    for _ in range(steps):
        fan = 0
        mist = 0
        
        # 1. Get Control Output & Run Physics
        if controller_type == 'Mamdani':
            # env.step() handles physics internally
            t, fan, _ = env.step(external_heat, 'Vegetative')
            current_temp = t
            # Mist power is stored in env by our recent modification
            mist = getattr(env, 'current_mist_power', 0)
            
        elif controller_type == 'Sugeno':
            # Compute Sugeno Output
            fan, mist = sugeno_ctrl.compute(current_temp, current_hum, 5) # 5=Vegetative
            
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
        energy_acc += (fan + mist) # Sum of fan and mist power
        
    # Calculate Metrics
    avg_error = error_acc / steps
    
    # Response Time: Steps until error stays below 1.0 degree
    response_time = steps # Default if never reached
    for i, t in enumerate(temp_history):
        if abs(t - target_temp) < 1.0:
            # Check if it stays close (simple check: next 3 steps are also close)
            if i + 3 < len(temp_history):
                if all(abs(temp_history[j] - target_temp) < 1.5 for j in range(i, i+3)):
                    response_time = i
                    break
            else:
                response_time = i
                break
            
    # Smoothness: Standard Deviation of Fan Output (lower is smoother)
    smoothness = np.std(fan_history)
    
    return response_time, avg_error, energy_acc, smoothness

def run_comparison():
    print("\n=== CONTROLLER COMPARISON SIMULATION ===")
    
    env = GreenhouseEnvironment()
    sugeno = SugenoController()
    
    # Disable RL exploration for the comparison
    env.epsilon = 0.0 
    
    # 1. Generate Scenarios (Fairness: Same scenarios for both)
    scenarios = []
    for _ in range(20):
        scenarios.append({
            'ext_heat': np.random.uniform(0.5, 4.0),
            'target': np.random.choice([18, 24, 28])
        })
        
    # 2. Run Mamdani Loop
    print("Running Mamdani Simulation Loop...")
    mamdani_results = {'rt': [], 'err': [], 'en': [], 'sm': []}
    for sc in scenarios:
        rt, err, en, sm = run_simulation_episode('Mamdani', env, sugeno, sc['ext_heat'], sc['target'])
        mamdani_results['rt'].append(rt)
        mamdani_results['err'].append(err)
        mamdani_results['en'].append(en)
        mamdani_results['sm'].append(sm)
        
    # 3. Run Sugeno Loop
    print("Running Sugeno Simulation Loop...")
    sugeno_results = {'rt': [], 'err': [], 'en': [], 'sm': []}
    for sc in scenarios:
        rt, err, en, sm = run_simulation_episode('Sugeno', env, sugeno, sc['ext_heat'], sc['target'])
        sugeno_results['rt'].append(rt)
        sugeno_results['err'].append(err)
        sugeno_results['en'].append(en)
        sugeno_results['sm'].append(sm)
        
    # 4. Output Results Table
    print("\n" + "="*85)
    print(f"{'Controller Type':<20} | {'Avg Response Time':<18} | {'Avg Error':<10} | {'Energy Usage':<12} | {'Smoothness Score':<15}")
    print("-" * 85)
    
    # Calculate Averages
    m_rt = np.mean(mamdani_results['rt'])
    m_err = np.mean(mamdani_results['err'])
    m_en = np.mean(mamdani_results['en'])
    m_sm = np.mean(mamdani_results['sm'])
    
    s_rt = np.mean(sugeno_results['rt'])
    s_err = np.mean(sugeno_results['err'])
    s_en = np.mean(sugeno_results['en'])
    s_sm = np.mean(sugeno_results['sm'])
    
    print(f"{'Mamdani':<20} | {m_rt:<18.2f} | {m_err:<10.2f} | {m_en:<12.1f} | {m_sm:<15.2f}")
    print(f"{'Sugeno':<20} | {s_rt:<18.2f} | {s_err:<10.2f} | {s_en:<12.1f} | {s_sm:<15.2f}")
    print("="*85 + "\n")

if __name__ == "__main__":
    run_comparison()
