import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class GreenhouseEnvironment:
    def __init__(self, target_temp=24.0):
        self.target_temp = target_temp
        self.current_temp = 20.0
        self.current_hum = 50.0
        self.iteration = 0
        
        self.temp_in = ctrl.Antecedent(np.arange(0, 51, 1), 'temperature')
        self.hum_in = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
        self.fan_out = ctrl.Consequent(np.arange(0, 101, 1), 'fan_power')
        self.mist_out = ctrl.Consequent(np.arange(0, 101, 1), 'mist_intensity')

        self.temp_in.automf(5, names=['freezing', 'cold', 'optimal', 'warm', 'hot'])
        self.hum_in.automf(5, names=['dry', 'low', 'optimal', 'high', 'saturated'])
        self.fan_out.automf(5, names=['off', 'low', 'medium', 'high', 'max'])
        self.mist_out.automf(3, names=['off', 'medium', 'high'])

        rule1 = ctrl.Rule(self.temp_in['hot'] | self.temp_in['warm'], self.fan_out['max'])
        rule2 = ctrl.Rule(self.temp_in['optimal'], self.fan_out['low'])
        rule3 = ctrl.Rule(self.temp_in['cold'], self.fan_out['off'])
        
        rule4 = ctrl.Rule(self.hum_in['dry'], self.mist_out['high'])
        rule5 = ctrl.Rule(self.hum_in['optimal'], self.mist_out['medium'])
        rule6 = ctrl.Rule(self.hum_in['saturated'] | self.hum_in['high'], self.mist_out['off'])

        
        self.ctrl_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
        self.simulation = ctrl.ControlSystemSimulation(self.ctrl_system)

        self.q_table = np.zeros((20, 3)) 
        self.epsilon = 0.1
        self.alpha = 0.5
        self.gamma = 0.9

    def get_state_index(self, error):
        clamped_error = max(-10, min(10, error))
        return int(clamped_error + 10) 

    def step(self, external_temp_influence):
        """ Runs one time step of the simulation """
        
        error = self.current_temp - self.target_temp
        state_idx = self.get_state_index(error)

        self.simulation.input['temperature'] = self.current_temp
        self.simulation.input['humidity'] = self.current_hum
        
        try:
            self.simulation.compute()
            base_fan_power = self.simulation.output['fan_power']
        except:
            base_fan_power = 0

        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice([0, 1, 2])
        else:
            action = np.argmax(self.q_table[state_idx])

        adjustment = 0
        if action == 1: adjustment = 10
        if action == 2: adjustment = -10
        
        final_fan_power = max(0, min(100, base_fan_power + adjustment))

        cooling_effect = (final_fan_power / 20.0) 
        warming_effect = external_temp_influence
        
        prev_temp = self.current_temp
        self.current_temp += (warming_effect - cooling_effect)

        new_error = abs(self.current_temp - self.target_temp)
        reward = -new_error

        new_state_idx = self.get_state_index(self.current_temp - self.target_temp)
        old_value = self.q_table[state_idx, action]
        next_max = np.max(self.q_table[new_state_idx])
        
        self.q_table[state_idx, action] = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

        return self.current_temp, final_fan_power, reward