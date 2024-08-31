import numpy as np
import matplotlib.pyplot as plt

# Parameters
charging_power = 50  # kW
battery_capacity = 200  # kWh
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

# Vehicle plugged-in times (1 if plugged in, 0 if not)
plugged_in_schedule = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# Initialize variables
battery_level = np.zeros(time_horizon + 1)
battery_level[0] = initial_battery_level
charging_schedule = np.zeros(time_horizon)
total_cost = 0

# Iterative charging process
for t in range(time_horizon):
    if plugged_in_schedule[t] == 1 and battery_level[t] < target_battery_level:
        # Calculate the possible charging power considering efficiency and battery capacity
        possible_charging_power = min(charging_power, 
                                      (target_battery_level - battery_level[t]) / charging_efficiency)
        # Update battery level
        battery_level[t + 1] = battery_level[t] + possible_charging_power * charging_efficiency
        # Record the charging power
        charging_schedule[t] = possible_charging_power
        # Calculate the cost
        total_cost += possible_charging_power * electricity_prices[t]
    else:
        # No charging
        battery_level[t + 1] = battery_level[t]

# Plot results
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(time_steps, electricity_prices, label='Electricity Prices', color='orange')
plt.ylabel('Price ($/kWh)')
plt.legend(loc='upper right')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.step(time_steps, plugged_in_schedule, label='Plugged-in Schedule', where='post', color='blue')
plt.ylabel('Plugged In (1=Yes, 0=No)')
plt.legend(loc='upper left')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.step(time_steps, battery_level[:-1], label='Battery Level', where='post')
plt.xlabel('Time (hours)')
plt.ylabel('Battery Level (kWh)')
plt.legend(loc='upper left')
plt.grid(True)

plt.tight_layout()
plt.show()

print("Charging Schedule (kW):", charging_schedule)
print("Final Battery Level (kWh):", battery_level[-1])
print("Total Charging Cost ($):", total_cost)
