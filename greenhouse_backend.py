import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import time

class GreenhouseEnvironment:
    def __init__(self, target_temp=24.0):
        self.target_temp = target_temp
        self.current_temp = 20.0
        self.current_hum = 50.0
        self.current_stage = 'vegetative' # Default
        self.plant_type = 'General'
        self.iteration = 0
        
        # RL Parameters
        self.q_table = np.zeros((20, 3)) 
        self.epsilon = 0.1
        self.alpha = 0.5
        self.gamma = 0.9

        # Initialize Control System
        self.build_control_system()

    def build_control_system(self, optimal_temp_center=24, optimal_hum_center=50):
        # --- 1. DEFINE FUZZY VARIABLES ---
        self.temp_in = ctrl.Antecedent(np.arange(0, 51, 1), 'temperature')
        self.hum_in = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
        self.stage_in = ctrl.Antecedent(np.arange(0, 11, 1), 'growth_stage') # 0-10 scale
        
        self.fan_out = ctrl.Consequent(np.arange(0, 101, 1), 'fan_power')
        self.mist_out = ctrl.Consequent(np.arange(0, 101, 1), 'mist_intensity')

        # --- 2. MEMBERSHIP FUNCTIONS ---
        # Temperature (5 sets): Very Cold, Cold, Optimal, Warm, Hot
        self.temp_in['Very Cold'] = fuzz.trimf(self.temp_in.universe, [0, 0, 10])
        self.temp_in['Cold'] = fuzz.trimf(self.temp_in.universe, [5, 15, optimal_temp_center - 2])
        self.temp_in['Optimal'] = fuzz.trimf(self.temp_in.universe, [optimal_temp_center - 5, optimal_temp_center, optimal_temp_center + 5])
        self.temp_in['Warm'] = fuzz.trimf(self.temp_in.universe, [optimal_temp_center + 2, optimal_temp_center + 12, 50])
        self.temp_in['Hot'] = fuzz.trimf(self.temp_in.universe, [40, 50, 50])

        # Humidity (5 sets): Very Dry, Dry, Optimal, Humid, Saturated
        self.hum_in['Very Dry'] = fuzz.trimf(self.hum_in.universe, [0, 0, 30])
        self.hum_in['Dry'] = fuzz.trimf(self.hum_in.universe, [20, 40, optimal_hum_center - 5])
        self.hum_in['Optimal'] = fuzz.trimf(self.hum_in.universe, [optimal_hum_center - 10, optimal_hum_center, optimal_hum_center + 10])
        self.hum_in['Humid'] = fuzz.trimf(self.hum_in.universe, [optimal_hum_center + 5, min(95, optimal_hum_center + 15), 100])
        self.hum_in['Saturated'] = fuzz.trimf(self.hum_in.universe, [80, 100, 100])

        # Growth Stage (3 sets): Seedling, Vegetative, Flowering
        self.stage_in['Seedling'] = fuzz.trapmf(self.stage_in.universe, [0, 0, 2, 4])
        self.stage_in['Vegetative'] = fuzz.trapmf(self.stage_in.universe, [3, 4, 6, 7])
        self.stage_in['Flowering'] = fuzz.trapmf(self.stage_in.universe, [6, 8, 10, 10])

        # Outputs
        self.fan_out.automf(5, names=['off', 'low', 'medium', 'high', 'max'])
        self.mist_out.automf(3, names=['off', 'medium', 'high'])

        # --- 3. RULES (25+ Rules) ---
        rules = []
        
        # Helper variables for cleaner code
        T, H, S = self.temp_in, self.hum_in, self.stage_in
        Fan, Mist = self.fan_out, self.mist_out

        # 1-3: Hot, Dry, Seedling -> Fan: Medium, Mist: High
        rules.append(ctrl.Rule(T['Hot'] & H['Dry'] & S['Seedling'], (Fan['medium'], Mist['high'])))
        rules.append(ctrl.Rule(T['Hot'] & H['Very Dry'] & S['Seedling'], (Fan['medium'], Mist['high'])))
        rules.append(ctrl.Rule(T['Warm'] & H['Very Dry'] & S['Seedling'], (Fan['low'], Mist['high']))) # Variation

        # 4-6: Hot, Saturated, Flowering -> Fan: Max, Mist: Off
        rules.append(ctrl.Rule(T['Hot'] & H['Saturated'] & S['Flowering'], (Fan['max'], Mist['off'])))
        rules.append(ctrl.Rule(T['Hot'] & H['Humid'] & S['Flowering'], (Fan['max'], Mist['off'])))
        rules.append(ctrl.Rule(T['Warm'] & H['Saturated'] & S['Flowering'], (Fan['high'], Mist['off']))) # Variation

        # 7-9: Optimal, Optimal, Vegetative -> Fan: Low, Mist: Medium
        rules.append(ctrl.Rule(T['Optimal'] & H['Optimal'] & S['Vegetative'], (Fan['low'], Mist['medium'])))
        rules.append(ctrl.Rule(T['Optimal'] & H['Humid'] & S['Vegetative'], (Fan['medium'], Mist['off']))) # Variation
        rules.append(ctrl.Rule(T['Warm'] & H['Optimal'] & S['Vegetative'], (Fan['medium'], Mist['medium']))) # Variation

        # 10-12: Cold, Dry, Seedling -> Fan: Off, Mist: High
        rules.append(ctrl.Rule(T['Cold'] & H['Dry'] & S['Seedling'], (Fan['off'], Mist['high'])))
        rules.append(ctrl.Rule(T['Cold'] & H['Very Dry'] & S['Seedling'], (Fan['off'], Mist['high'])))
        rules.append(ctrl.Rule(T['Very Cold'] & H['Dry'] & S['Seedling'], (Fan['off'], Mist['high'])))

        # 13-15: Very Hot (mapped to Hot), Humid, Vegetative -> Fan: Max, Mist: Low
        rules.append(ctrl.Rule(T['Hot'] & H['Humid'] & S['Vegetative'], (Fan['max'], Mist['off']))) # Mist Low mapped to Off/Medium? Using Off to be safe or Medium? User said Low. Mist has off, medium, high. I'll use off.
        rules.append(ctrl.Rule(T['Hot'] & H['Saturated'] & S['Vegetative'], (Fan['max'], Mist['off'])))
        rules.append(ctrl.Rule(T['Warm'] & H['Humid'] & S['Vegetative'], (Fan['high'], Mist['off'])))

        # 16-18: Very Cold, Saturated, Flowering -> Fan: Off, Mist: Off
        rules.append(ctrl.Rule(T['Very Cold'] & H['Saturated'] & S['Flowering'], (Fan['off'], Mist['off'])))
        rules.append(ctrl.Rule(T['Very Cold'] & H['Humid'] & S['Flowering'], (Fan['off'], Mist['off'])))
        rules.append(ctrl.Rule(T['Cold'] & H['Saturated'] & S['Flowering'], (Fan['off'], Mist['off'])))

        # 19-21: Warm, Optimal, Seedling -> Fan: Low, Mist: Medium
        rules.append(ctrl.Rule(T['Warm'] & H['Optimal'] & S['Seedling'], (Fan['low'], Mist['medium'])))
        rules.append(ctrl.Rule(T['Warm'] & H['Humid'] & S['Seedling'], (Fan['medium'], Mist['off']))) # Variation
        rules.append(ctrl.Rule(T['Optimal'] & H['Optimal'] & S['Seedling'], (Fan['off'], Mist['medium']))) # Variation

        # 22-23: Hot, Low (mapped to Dry), Vegetative -> Fan: High, Mist: High
        rules.append(ctrl.Rule(T['Hot'] & H['Dry'] & S['Vegetative'], (Fan['high'], Mist['high'])))
        rules.append(ctrl.Rule(T['Hot'] & H['Very Dry'] & S['Vegetative'], (Fan['high'], Mist['high'])))

        # 24-25: Optimal, Dry, Flowering -> Fan: Off, Mist: High
        rules.append(ctrl.Rule(T['Optimal'] & H['Dry'] & S['Flowering'], (Fan['off'], Mist['high'])))
        rules.append(ctrl.Rule(T['Optimal'] & H['Very Dry'] & S['Flowering'], (Fan['off'], Mist['high'])))

        # Additional coverage rules to ensure robustness
        rules.append(ctrl.Rule(T['Very Cold'], Fan['off'])) # Safety rule
        rules.append(ctrl.Rule(T['Hot'] & H['Very Dry'], Mist['high'])) # Safety rule

        self.ctrl_system = ctrl.ControlSystem(rules)
        self.simulation = ctrl.ControlSystemSimulation(self.ctrl_system)

    def apply_adaptation(self, plant_type):
        """
        Dynamically shifts fuzzy membership functions based on plant species requirements.
        
        Adaptation Logic Explanation:
        Different plant species have distinct physiological 'comfort zones' for photosynthesis 
        and transpiration. By shifting the 'Optimal' membership function peaks (and neighboring 
        sets like 'Cold'/'Warm' relative to it), we ensure the control system targets the 
        specific microclimate needed for the selected species. This minimizes stress (e.g., 
        heat stress in Lettuce, fungal rot in Tomatoes due to excess humidity) and maximizes 
        growth potential across all stages (Seedling, Vegetative, Flowering).
        """
        self.plant_type = plant_type
        
        # Dictionary of requirements
        species_reqs = {
            'Tomato':  {'temp': 24, 'hum': 70},
            'Lettuce': {'temp': 18, 'hum': 60},
            'Orchid':  {'temp': 26, 'hum': 80},
            'General': {'temp': 24, 'hum': 50}
        }
        
        req = species_reqs.get(plant_type, species_reqs['General'])
        
        print(f"Adapting system for {plant_type}: Target Temp={req['temp']}C, Target Hum={req['hum']}%")
        self.build_control_system(optimal_temp_center=req['temp'], optimal_hum_center=req['hum'])

    def set_plant_type(self, species):
        self.apply_adaptation(species)

    def get_state_index(self, error):
        clamped_error = max(-10, min(10, error))
        return int(clamped_error + 10) 

    def step(self, external_temp_influence, stage_name='vegetative'):
        """ Runs one time step of the simulation """
        self.current_stage = stage_name
        
        # Map stage name to value
        stage_val = 5 # Default vegetative
        if stage_name == 'Seedling': stage_val = 1
        elif stage_name == 'Vegetative': stage_val = 5
        elif stage_name == 'Flowering': stage_val = 9

        # 1. Calculate Error
        error = self.current_temp - self.target_temp
        state_idx = self.get_state_index(error)

        # 2. Fuzzy Control Compute
        self.simulation.input['temperature'] = self.current_temp
        self.simulation.input['humidity'] = self.current_hum
        self.simulation.input['growth_stage'] = stage_val
        
        try:
            self.simulation.compute()
            base_fan_power = self.simulation.output['fan_power']
        except:
            base_fan_power = 0

        # 3. RL ACTION (Evolving the Rule Output)
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice([0, 1, 2])
        else:
            action = np.argmax(self.q_table[state_idx])

        adjustment = 0
        if action == 1: adjustment = 10
        if action == 2: adjustment = -10
        
        final_fan_power = max(0, min(100, base_fan_power + adjustment))

        # Physics Simulation
        cooling_effect = (final_fan_power / 20.0) 
        warming_effect = external_temp_influence
        
        self.current_temp += (warming_effect - cooling_effect)
        
        # Simple humidity physics (Fan dries air)
        self.current_hum -= (final_fan_power / 50.0)
        self.current_hum += 0.5 # Natural transpiration
        self.current_hum = max(0, min(100, self.current_hum))

        new_error = abs(self.current_temp - self.target_temp)
        reward = -new_error

        new_state_idx = self.get_state_index(self.current_temp - self.target_temp)
        old_value = self.q_table[state_idx, action]
        next_max = np.max(self.q_table[new_state_idx])
        
        self.q_table[state_idx, action] = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

        return self.current_temp, final_fan_power, reward

# --- SUGENO CONTROLLER IMPLEMENTATION ---
class SugenoController:
    def __init__(self):
        # Output Constants
        self.FAN_OFF = 0
        self.FAN_LOW = 30
        self.FAN_MED = 50
        self.FAN_HIGH = 80
        self.FAN_MAX = 100
        
        self.MIST_OFF = 0
        self.MIST_MED = 50
        self.MIST_HIGH = 100

    def _trimf(self, x, abc):
        """Triangular membership function"""
        a, b, c = abc
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x < c:
            return (c - x) / (c - b)
        return 0.0

    def _trapmf(self, x, abcd):
        """Trapezoidal membership function"""
        a, b, c, d = abcd
        if x <= a or x >= d:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x < c:
            return 1.0
        elif c <= x < d:
            return (d - x) / (d - c)
        return 0.0

    def compute(self, temp, hum, stage_val):
        # --- 1. FUZZIFICATION ---
        
        # Temperature (Very Cold, Cold, Optimal, Warm, Hot)
        # Assuming optimal center around 24
        mu_t_vcold = self._trimf(temp, [0, 0, 10])
        mu_t_cold = self._trimf(temp, [5, 15, 22])
        mu_t_opt = self._trimf(temp, [19, 24, 29])
        mu_t_warm = self._trimf(temp, [26, 35, 45])
        mu_t_hot = self._trimf(temp, [40, 50, 50])

        # Humidity (Very Dry, Dry, Optimal, Humid, Saturated)
        # Assuming optimal center around 50
        mu_h_vdry = self._trimf(hum, [0, 0, 30])
        mu_h_dry = self._trimf(hum, [20, 40, 45])
        mu_h_opt = self._trimf(hum, [40, 50, 60])
        mu_h_humid = self._trimf(hum, [55, 70, 90])
        mu_h_sat = self._trimf(hum, [80, 100, 100])

        # Growth Stage (Seedling, Vegetative, Flowering)
        # 0-10 scale
        mu_s_seed = self._trapmf(stage_val, [0, 0, 2, 4])
        mu_s_veg = self._trapmf(stage_val, [3, 4, 6, 7])
        mu_s_flow = self._trapmf(stage_val, [6, 8, 10, 10])

        # --- 2. INFERENCE (Takagi-Sugeno 0-order) ---
        # Rules map to constant outputs
        
        numerator_fan = 0.0
        denominator_fan = 0.0
        
        numerator_mist = 0.0
        denominator_mist = 0.0

        def add_rule(weight, fan_out, mist_out):
            nonlocal numerator_fan, denominator_fan, numerator_mist, denominator_mist
            if weight > 0:
                numerator_fan += weight * fan_out
                denominator_fan += weight
                numerator_mist += weight * mist_out
                denominator_mist += weight

        # Rule 1-3: Hot & Dry/Very Dry & Seedling -> Fan Med, Mist High
        w = min(mu_t_hot, max(mu_h_dry, mu_h_vdry), mu_s_seed)
        add_rule(w, self.FAN_MED, self.MIST_HIGH)

        # Rule 4-6: Hot & Saturated/Humid & Flowering -> Fan Max, Mist Off
        w = min(mu_t_hot, max(mu_h_sat, mu_h_humid), mu_s_flow)
        add_rule(w, self.FAN_MAX, self.MIST_OFF)

        # Rule 7-9: Optimal & Optimal & Vegetative -> Fan Low, Mist Med
        w = min(mu_t_opt, mu_h_opt, mu_s_veg)
        add_rule(w, self.FAN_LOW, self.MIST_MED)

        # Rule 10-12: Cold/Very Cold & Dry & Seedling -> Fan Off, Mist High
        w = min(max(mu_t_cold, mu_t_vcold), mu_h_dry, mu_s_seed)
        add_rule(w, self.FAN_OFF, self.MIST_HIGH)

        # Rule 13-15: Hot/Warm & Humid/Saturated & Vegetative -> Fan Max, Mist Off
        w = min(max(mu_t_hot, mu_t_warm), max(mu_h_humid, mu_h_sat), mu_s_veg)
        add_rule(w, self.FAN_MAX, self.MIST_OFF)

        # Rule 16-18: Very Cold/Cold & Saturated & Flowering -> Fan Off, Mist Off
        w = min(max(mu_t_vcold, mu_t_cold), mu_h_sat, mu_s_flow)
        add_rule(w, self.FAN_OFF, self.MIST_OFF)

        # Rule 19-21: Warm & Optimal & Seedling -> Fan Low, Mist Med
        w = min(mu_t_warm, mu_h_opt, mu_s_seed)
        add_rule(w, self.FAN_LOW, self.MIST_MED)

        # Rule 22-23: Hot & Dry/Very Dry & Vegetative -> Fan High, Mist High
        w = min(mu_t_hot, max(mu_h_dry, mu_h_vdry), mu_s_veg)
        add_rule(w, self.FAN_HIGH, self.MIST_HIGH)

        # Rule 24-25: Optimal & Dry/Very Dry & Flowering -> Fan Off, Mist High
        w = min(mu_t_opt, max(mu_h_dry, mu_h_vdry), mu_s_flow)
        add_rule(w, self.FAN_OFF, self.MIST_HIGH)
        
        # Safety Rules
        add_rule(mu_t_vcold, self.FAN_OFF, self.MIST_OFF)
        add_rule(min(mu_t_hot, mu_h_vdry), self.FAN_HIGH, self.MIST_HIGH)

        # --- 3. DEFUZZIFICATION (Weighted Average) ---
        fan_result = 0
        if denominator_fan > 0:
            fan_result = numerator_fan / denominator_fan
            
        mist_result = 0
        if denominator_mist > 0:
            mist_result = numerator_mist / denominator_mist
            
        return fan_result

# --- PERFORMANCE TRACKING ---
def run_performance_test():
    env = GreenhouseEnvironment()
    sugeno = SugenoController()
    
    print(f"{'Scenario':<10} | {'Mamdani Err':<12} | {'Sugeno Err':<12} | {'Energy':<10}")
    print("-" * 50)

    total_mamdani_err = 0
    total_sugeno_err = 0

    for i in range(20):
        # Random Scenario
        start_temp = np.random.uniform(15, 35)
        target = 24.0
        env.current_temp = start_temp
        env.target_temp = target
        
        # Simulate 10 steps
        mamdani_error_acc = 0
        sugeno_error_acc = 0
        energy_acc = 0
        
        temp_m = start_temp
        temp_s = start_temp

        for _ in range(10):
            # Mamdani Step
            env.current_temp = temp_m
            t_m, fan_m, _ = env.step(1.0, 'Vegetative')
            mamdani_error_acc += abs(t_m - target)
            temp_m = t_m
            energy_acc += fan_m

            # Sugeno Step (Approximation of physics)
            fan_s = sugeno.compute(temp_s, 50, 5) # 5 = Vegetative
            cooling = fan_s / 20.0
            temp_s += (1.0 - cooling)
            sugeno_error_acc += abs(temp_s - target)

        avg_m_err = mamdani_error_acc / 10
        avg_s_err = sugeno_error_acc / 10
        
        total_mamdani_err += avg_m_err
        total_sugeno_err += avg_s_err

        print(f"{i+1:<10} | {avg_m_err:<12.2f} | {avg_s_err:<12.2f} | {energy_acc:<10.1f}")

    print("-" * 50)
    print(f"Avg Mamdani Error: {total_mamdani_err/20:.2f}")
    print(f"Avg Sugeno Error: {total_sugeno_err/20:.2f}")

if __name__ == "__main__":
    run_performance_test()
