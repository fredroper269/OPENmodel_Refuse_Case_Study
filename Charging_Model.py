import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# Parameters
charging_power = 200  # kW
battery_capacity = 300  # kWh
initial_battery_level = 50  # kWh
target_battery_level = 180  # kWh
charging_efficiency = 0.95  # 95% efficiency

# Define the time horizon
time_horizon = 24  # hours
time_steps = np.arange(time_horizon)  # hourly time steps

# Example hourly electricity prices ($/kWh)
electricity_prices = np.array([0.134266923, 0.133265385, 0.117080769, 0.135543846, 0.137053846, 0.126766154, 
                               0.15458, 0.124686154, 0.128264615, 0.129893077, 0.146390769, 0.155038462, 0.181839231, 
                               0.196831538, 0.201834615, 0.210781538, 0.209139231, 0.21598, 0.229406923, 0.299633077, 
                               0.209276154, 0.180326923, 0.171807692, 0.152671538])

# Objective: Minimize the cost of charging
c = electricity_prices

# Constraints
A_eq = np.ones((1, time_horizon)) * charging_power * charging_efficiency
b_eq = target_battery_level - initial_battery_level

# Bounds: Charging power should be between 0 and the maximum charging power
bounds = [(0, charging_power)] * time_horizon

# Solve the linear programming problem
result = linprog(c, A_eq=A_eq, b_eq=[b_eq], bounds=bounds, method='highs')

# Extract the optimal charging schedule
optimal_charging_schedule = result.x

# Calculate the battery level over time
battery_level = np.cumsum(optimal_charging_schedule * charging_power * charging_efficiency) + initial_battery_level

# Plot results
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(time_steps, electricity_prices, label='Electricity Prices', color='orange')
plt.ylabel('Price ($/kWh)')
plt.legend(loc='upper right')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.step(time_steps, battery_level, label='Battery Level', where='post')
plt.xlabel('Time (hours)')
plt.ylabel('Battery Level (kWh)')
plt.legend(loc='upper left')
plt.grid(True)

plt.tight_layout()
plt.show()

print("Optimal Charging Schedule (kW):", optimal_charging_schedule)
print("Final Battery Level (kWh):", battery_level[-1])
print("Total Charging Cost ($):", np.dot(optimal_charging_schedule, electricity_prices))
