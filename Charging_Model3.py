import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_vehicles = 1  # Number of vehicles
charging_power_per_vehicle = 50  # kW per vehicle
battery_capacity_per_vehicle = 200  # kWh per vehicle
initial_battery_levels = np.random.uniform(50, 100, num_vehicles)  # kWh
target_battery_levels = np.full(num_vehicles, 180)  # kWh
charging_efficiency = 0.95  # 95% efficiency
max_system_power = 500  # kW, maximum power capacity of the system

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
battery_levels = np.zeros((num_vehicles, time_horizon + 1))
battery_levels[:, 0] = initial_battery_levels
charging_schedules = np.zeros((num_vehicles, time_horizon))
total_cost = 0

# Iterative charging process
for t in range(time_horizon):
    available_power = max_system_power  # Reset available power at each time step
    for v in range(num_vehicles):
        if plugged_in_schedule[t] == 1 and battery_levels[v, t] < target_battery_levels[v]:
            # Calculate the possible charging power considering efficiency and battery capacity
            possible_charging_power = min(charging_power_per_vehicle, 
                                          (target_battery_levels[v] - battery_levels[v, t]) / charging_efficiency)
            # Ensure that the total charging power does not exceed the available power
            actual_charging_power = min(possible_charging_power, available_power)
            # Update battery level
            battery_levels[v, t + 1] = battery_levels[v, t] + actual_charging_power * charging_efficiency
            # Record the charging power
            charging_schedules[v, t] = actual_charging_power
            # Reduce the available power
            available_power -= actual_charging_power
            # Calculate the cost
            total_cost += actual_charging_power * electricity_prices[t]
        else:
            # No charging
            battery_levels[v, t + 1] = battery_levels[v, t]

# Plot results for the first vehicle
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
for v in range(num_vehicles):
    plt.step(time_steps, battery_levels[v, :-1], label=f'Battery Level Vehicle {v+1}', where='post')
plt.xlabel('Time (hours)')
plt.ylabel('Battery Level (kWh)')
plt.legend(loc='upper left')
plt.grid(True)

plt.tight_layout()
plt.show()

print("Charging Schedules (kW):", charging_schedules)
print("Final Battery Levels (kWh):", battery_levels[:, -1])
print("Total Charging Cost ($):", total_cost)
