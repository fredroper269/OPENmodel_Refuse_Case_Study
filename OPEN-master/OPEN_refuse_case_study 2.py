# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:15:06 2024

@author: Scot Wheeler - University of Oxford

Basic demonstration of vehicle to grid optimisation
"""

#import modules
import os
from os.path import normpath, join
import copy
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import numpy as np
import picos as pic
import matplotlib.pyplot as plt
from datetime import date, timedelta
import matplotlib.dates as mdates
from datetime import datetime, timedelta

import System.Assets as AS
import System.Markets as MK
import System.EnergySystem as ES

import sys

# ignore future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

print('Code started.')
# plt.close('all')

############## VERSION ##############


__version__ = "0.0.1"

#######################################
###
# Case Study: Refuse V2G
###
#######################################

save_string = 'RefuseCaseStudyExample'
path_string = normpath('Results/RefuseV2G/')
if not os.path.isdir(path_string):
    os.makedirs(path_string)


#######################################
# STEP 0: Load raw Data
#######################################


input_data_path = os.path.join("Data", "Refuse_case(ToU).csv")
all_data = pd.read_csv(input_data_path)
#PVpu_raw_smr = all_data["PV"].to_numpy()
#Loads_raw_smr = all_data["Load"].to_numpy()
prices_import = all_data["Import_price"].to_numpy()
prices_export = all_data["Export_price"].to_numpy()


#######################################
# STEP 1: setup parameters
#######################################

# core parameters
dt = 30 / 60  # simulation time resolution
T = int(24 / dt)  # Number of intervals
dt_ems = 30 / 60  # EMS optimisation resolution - will have biggest impact on run time
T_ems = int(T * dt / dt_ems)  # Number of EMS intervals


# start time of simulation
T0 = 8  # start time (hour of day) of the model, T0=0  is midnight to midnight, T0=8 is 8am to 8am

# installed PV capacity (kW)
#Ppv_nom = 400  # power rating of the PV generation

# Electric Vehicle (EV) parameters
eff_EV = 0.99 # charging /discharging efficiency for simulation
eff_EV_opt = 0.99 # charging /discharging efficiency for optimisation
N_EVs = 1 # number of EVs
Emax_EV = 300  # maximum EV energy level (size of usable battery (kWh))
Emin_EV = 0  # minimum EV energy level 
P_max_EV = 200# maximum EV charging power (power of charger for charging)
P_min_EV = -200 # minimum EV charging power (power of charger for exporting e.g. V2G)

# EV arrival & departure times and energy levels on arrival
# random EV initial energy level - define as soc (0-1)
#E0_EVs = Emax_EV * np.random.uniform(0.1, 0.3, N_EVs)
E0_EVs = Emax_EV * np.ones(N_EVs) * 0.2

# random EV arrival times between 12:00 and 14:00 (was 16 and 18)
ta_EVs = np.random.randint(int(12 / dt_ems), int(14 / dt_ems), N_EVs) - int(T0 / dt_ems)
# random EV departure times between 4am and 5am (add 24 for next day)
td_EVs = np.random.randint(int(28 / dt_ems), int(29 / dt_ems), N_EVs) - int(T0 / dt_ems)

# uncomment below to set all to the exact same initial level and times
# E0_EVs = Emax_EV * np.ones(N_EVs) * 0.2
# ta_EVs = (np.ones(N_EVs,dtype=np.int32) * int(16 / dt_ems)) - int(T0 / dt_ems)
# td_EVs = (np.ones(N_EVs,dtype=np.int32) * int(29 / dt_ems)) - int(T0 / dt_ems)

# Ensure EVs can be feasibility charged
for i in range(N_EVs):
    td_EVs[i] = np.max([td_EVs[i], ta_EVs[i]])
    E0_EVs[i] = np.max([E0_EVs[i], Emax_EV - P_max_EV * (td_EVs[i] - ta_EVs[i])])

# Market parameters
dt_market = dt_ems  # market and EMS have the same time-series
T_market = T_ems  # market and EMS have same length

# uncomment below to define tariffs here:
# prices_export = np.hstack((0.04 * np.ones(int(T_market * 16 / 24)),
#                           0.15 * np.ones(int(T_market * 2 / 24)),
#                           0.04 * np.ones(int(T_market * 6 / 24))
#                             ))  # money received of net exports
# prices_import = np.hstack((0.07 * np.ones(int(T_market * 7 / 24)),
#                           0.15 * np.ones(int(T_market * 2 / 24)),
#                           0.10 * np.ones(int(T_market * 7 / 24)),
#                           0.20 * np.ones(int(T_market * 2 / 24)),
#                           0.07 * np.ones(int(T_market * 6 / 24)),
#                             ))  # price of net imports


demand_charge = 0.0  # price per kW for the maximum demand #Add a price here for cost /kW for connection

# use this to define the maximum allowable import / export power to the site
# this may become a limit if looking at high power EV or large numbers of EVs
Pmax_market = 5000 * np.ones(T_market)  # maximum import power #total size of connection (was at 5000)
Pmin_market = -5000 * np.ones(T_market)  # maximum export power (was at -5000)

#######################################
# STEP 1b: wrangle input data
#######################################
# convert to same time resolution as defined by dt
# PVpu_conv = np.array([np.interp(np.arange(0, len(PVpu_raw_smr[:, pv]), PVpu_raw_smr.shape[0] / T), np.arange(0, len(PVpu_raw_smr[:, pv])), PVpu_raw_smr[:, pv]) for pv in range(PVpu_raw_smr.shape[1])]).T
# Loads_conv = np.array([np.interp(np.arange(0, len(Loads_raw_smr[:, l]), Loads_raw_smr.shape[0] / T), np.arange(0, len(Loads_raw_smr[:, l])), Loads_raw_smr[:, l]) for l in range(Loads_raw_smr.shape[1])]).T

# shift tariff data by T0
prices_import = np.concatenate((prices_import[int(T0 / dt_ems):], prices_import[:int(T0 / dt_ems)]))
prices_export = np.concatenate((prices_export[int(T0 / dt_ems):], prices_export[:int(T0 / dt_ems)]))


# shift pv and load data by T0
#PVpu = np.concatenate((PVpu_raw_smr[int(T0 / dt_ems):], PVpu_raw_smr[:int(T0 / dt_ems)]))
#Loads = np.concatenate((Loads_raw_smr[int(T0 / dt_ems):], Loads_raw_smr[:int(T0 / dt_ems)]))

# if you want zero load, uncomment line below
# Loads = np.zeros(Loads.shape)

# if you want zero PV, uncomment line below
# PVpu = np.zeros(PVpu.shape)


#######################################
# STEP 2: setup the network
#######################################

# (from https://github.com/e2nIEE/pandapower/blob/master/tutorials/minimal_example.ipynb)
network = pp.create_empty_network()
# create buses
bus1 = pp.create_bus(network, vn_kv=20., name="bus 1")
bus2 = pp.create_bus(network, vn_kv=0.4, name="bus 2")
bus3 = pp.create_bus(network, vn_kv=0.4, name="bus 3")
# create bus elements
pp.create_ext_grid(network, bus=bus1, vm_pu=1.0, name="Grid Connection")
# create branch elements
trafo = pp.create_transformer(network, hv_bus=bus1, lv_bus=bus2, std_type="0.4 MVA 20/0.4 kV", name="Trafo") #Transformer rating
line = pp.create_line(network, from_bus=bus2, to_bus=bus3, length_km=0.1, std_type="NAYY 4x50 SE", name="Line")
N_buses = network.bus['name'].size


#######################################
# STEP 3: setup the assets
#######################################

# initiate empty lists for different types of assets
storage_assets = []
nondispatch_assets = []

# PV source at bus 3
#Pnet = -PVpu * Ppv_nom  # 100kW PV plant
#Qnet = np.zeros(T)
#PV_gen_bus3 = AS.NondispatchableAsset(Pnet, Qnet, bus3, dt, T)
#nondispatch_assets.append(PV_gen_bus3)

# Load at bus 3
#Pnet = Loads#np.sum(Loads, 1)  # summed load across 120 households
#Qnet = np.zeros(T)
#load_bus3 = AS.NondispatchableAsset(Pnet, Qnet, bus3, dt, T)
#nondispatch_assets.append(load_bus3)

# EVs at bus 634
for i in range(N_EVs):
    Emax_ev_i = Emax_EV * np.ones(T_ems)
    Emin_ev_i = Emin_EV * np.ones(T_ems)
    Pmax_ev_i = np.zeros(T_ems)
    Pmin_ev_i = np.zeros(T_ems)
    for t in range(ta_EVs[i], int(min(td_EVs[i], T_ems))):
        Pmax_ev_i[t] = P_max_EV
        Pmin_ev_i[t] = P_min_EV
    bus_id_ev_i = bus3
    ev_i = AS.StorageAsset(Emax_ev_i, Emin_ev_i, Pmax_ev_i, Pmin_ev_i,
                           E0_EVs[i], Emax_EV, bus_id_ev_i, dt, T, dt_ems,
                           T_ems, Pmax_abs=P_max_EV, c_deg_lin=0,
                           eff=eff_EV, eff_opt=eff_EV_opt)
    storage_assets.append(ev_i)

#######################################
# STEP 4: setup the market
#######################################

bus_id_market = bus1
market = MK.Market(bus_id_market, prices_export, prices_import, demand_charge, Pmax_market, Pmin_market, dt_market, T_market)

#######################################
# STEP 5: setup the energy system
#######################################

energy_system = ES.EnergySystem(storage_assets, nondispatch_assets, network, market, dt, T, dt_ems, T_ems)

#######################################
# STEP 6: simulate the energy system:
#######################################

output = energy_system.simulate_network()
# output = energy_system.EMS_copper_plate()
# output = energy_system.simulate_network_bldg()

# =============================================================================
# Retrieve outputs
# =============================================================================

buses_Vpu = output['buses_Vpu']
buses_Vang = output['buses_Vang']
buses_Pnet = output['buses_Pnet']
buses_Qnet = output['buses_Qnet']
Pnet_market = output['Pnet_market']
Qnet_market = output['Qnet_market']
buses_Vpu = output['buses_Vpu']
P_import_ems = output['P_import_ems']
P_export_ems = output['P_export_ems']
P_demand_ems = output['P_demand_ems']
P_ES_ems = output['P_ES_ems']


P_net_base = np.zeros(T)
P_demand_base = np.zeros(T)
P_gen_base = np.zeros(T)
for i in range(len(nondispatch_assets)):
    bus_id = nondispatch_assets[i].bus_id
    P_net_base += nondispatch_assets[i].Pnet
    if nondispatch_assets[i].Pnet.mean()>0:
        P_demand_base += nondispatch_assets[i].Pnet
    else:
        P_gen_base += nondispatch_assets[i].Pnet
    
# P_demand_base_pred = np.zeros(T)
# for i in range(len(nondispatch_assets)):
#     bus_id = nondispatch_assets[i].bus_id
#     P_demand_base_pred += nondispatch_assets[i].Pnet_pred
    
# P_demand_base_pred_ems = np.zeros(T_ems)
# for t_ems in range(T_ems):
#     t_indexes = (t_ems * dt_ems / dt + np.arange(0, dt_ems / dt)).astype(int)
#     P_demand_base_pred_ems[t_ems] = np.mean(P_demand_base_pred[t_indexes])

storage_asset_soc = pd.DataFrame(np.array([storage_asset.E / (Emax_EV - Emin_EV) for storage_asset in storage_assets]).T)
storage_asset_Pnet = pd.DataFrame(np.array([storage_asset.Pnet for storage_asset in storage_assets]).T)


#######################################
# STEP 7: plot results
#######################################

# Print revenue generated
revenue = market.calculate_revenue(Pnet_market, dt)
print('Net Revenue: £ ' + str(revenue))

# energy cost
energy_cost = market.calculate_revenue(Pnet_market, dt)
energy_cost_string = 'Total energy cost: £ %.2f' % (-1 * energy_cost)
print(energy_cost_string)

N_ESs = len(storage_assets)  # number of storage assets

EVs_tot = sum(P_ES_ems[:, n] for n in range(N_ESs))
# P_compare = P_demand_base_pred_ems + EVs_tot

# x-axis time values
base_time = datetime(2023, 7, 10)  # Starting date (can be any date)
time = [base_time + timedelta(hours=hr) for hr in (dt * np.arange(T) + T0)]
time_ems = [base_time + timedelta(hours=hr) for hr in (dt_ems * np.arange(T_ems) + T0)]
timeE = [base_time + timedelta(hours=hr) for hr in (dt * np.arange(T + 1) + T0)]


# %% plot base net load (load and pv) and total net power
fig, ax = plt.subplots(figsize=(6, 4), dpi=150, facecolor='w', edgecolor='k')
ax.set_title(label="Base load and total net power flow")
ax.fill_between(time, P_gen_base, step='post', alpha=0.2, color='g', label='Fixed Generation')
ax.fill_between(time, P_demand_base, step='post', alpha=0.2, color='k', label='Fixed Demand')
# ax.step(time, P_net_base, '--', label='Net Demand Base', where='post')
ax.bar([t+timedelta(minutes=7.5) for t in time], storage_asset_Pnet.sum(axis=1),
       color='indigo', width = timedelta(minutes=15), align='edge',
       label="EV charging (+ive) / discharging (-ive)", alpha=0.7)
ax.step(time, Pnet_market, label='Total Net demand', where='post', color='darkblue')
ax.set_ylabel('Power (kW)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.set_xlabel('Time (hh:mm)')
ax.set_xlim(min(time), max(time))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=2)
plt.grid(False)
plt.tight_layout()

# %% plot base net load (load and pv) and total net power with prices
fig, ax = plt.subplots(figsize=(6, 4), dpi=150, facecolor='w', edgecolor='k')
ax.set_title(label="Net load and tariff")
ax.fill_between(time, P_net_base, label='Fixed Net Load', step='post', color='grey', alpha=0.3)
ax.step(time, Pnet_market, label='Total Net Demand', where='post', color='darkblue')
ax.set_ylabel('Power (kW)')
# plt.ylim(500, 2100)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.set_xlabel('Time (hh:mm)')
ax.set_xlim(min(time), max(time))

ax2 = ax.twinx()
ax2.step(time_ems, prices_import, where='post', linestyle='--',
         linewidth=0.7, color='k', label='import rate')
ax2.step(time_ems, -1 * prices_export, where='post', linestyle='--',
         linewidth=0.7, color='g', label='export rate')
ax2.set_ylabel('Market price (£/kWh)')

plt.grid(False)
# Combine legends from both axes
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=2)
plt.tight_layout()
ax = plt.gca()
plt.savefig(join(path_string, normpath('P_actual_' + str(save_string) + '.pdf')),
            bbox_inches='tight')

# %% plot individual battery power flows
fig, ax = plt.subplots(figsize=(6, 4), dpi=150, facecolor='w', edgecolor='k')
ax.set_title(label="Individual EV power flow")
# plt.plot(time,sum(storage_assets[i].Pnet for i in range(N_ESs)))

for i in range(N_EVs):
    ax.step(time, storage_assets[i].Pnet, where='post')
# ax.axhline(0, color='k')
# ax.set_xlim(0, 24)
# plt.ylim(0, 10)
ax2 = ax.twinx()
ax2.step(time_ems, prices_import, where='post', linestyle='--', linewidth=0.7, color='k')
ax2.step(time_ems, -1 * prices_export, where='post', linestyle='--', linewidth=0.7, color='g')
# ax2.axhline(0, color='k')
ax2.set_ylabel('Import and export price (£/kWh)')
ax.set_ylabel('Power (kW)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.set_xlabel('Time (hh:mm)')
ax.set_xlim(min(time), max(time))
plt.grid(False)
ax = plt.gca()
plt.tight_layout()
plt.savefig(join(path_string, normpath('P_EVs_' + str(save_string) + '.pdf')),
            bbox_inches='tight')


# %% plot average battery soc
fig, ax = plt.subplots(figsize=(6, 4), dpi=150, facecolor='w', edgecolor='k')
ax.set_title(label="Average battery State of Charge")
ax.step(timeE, storage_asset_soc.mean(axis=1), where='post', label="Avg SoC", color='darkorange')
ax.set_ylabel('Average EV SOC')

ax.set_ylim(0, 1.1)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.set_xlabel('Time (hh:mm)')
ax.set_xlim(min(time), max(time))

ax2 = ax.twinx()
ax2.step(time_ems, prices_import, where='post', linestyle='--',
         linewidth=0.7, color='k', label='import rate')
ax2.step(time_ems, -1 * prices_export, where='post', linestyle='--',
         linewidth=0.7, color='g', label='export rate')
ax2.set_ylabel('Market price (£/kWh)')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=2)


plt.grid(False)
plt.tight_layout()
plt.savefig(join(path_string, normpath('E_EVs_' + str(save_string) + '.pdf')),
            bbox_inches='tight')


#%% save the data
# if x == "open_loop":
#     pickled_data_OL = (N_EVs, P_demand_base_pred_ems, P_compare, P_demand_base,\
#                     Pnet_market, storage_assets, N_ESs, nondispatch_assets,\
#                     time_ems, time, timeE, buses_Vpu)
#     pickle.dump(pickled_data_OL, open(join(path_string, normpath("EV_case_data_open_loop.p")), "wb"))

# if x == "mpc":
#     pickled_data_MPC = (N_EVs, P_demand_base_pred_ems, P_compare, P_demand_base,\
#                     Pnet_market, storage_assets, N_ESs, nondispatch_assets,\
#                     time_ems, time, timeE, buses_Vpu)
#     pickle.dump(pickled_data_MPC, open(join(path_string, normpath("EV_case_data_mpc.p")), "wb"))


#%%Printing the input and output variables (Additional).

from matplotlib.backends.backend_pdf import PdfPages
# Assuming you have the same data as previously provided
dt = 30/60
T = 24
dt_ems = 30/60
T_ems = 24
T0 = 8
Ppv_nom = 400

N_EVs = 3
Emax_EV = 300
Emin_EV = 0
P_max_EV = 300
P_min_EV = -300
ta_EVs = [12,13]
td_EVs = [28,29]

# Function to save a dataframe as a table figure
def add_df_to_pdf(df, title, pdf_pages):
    fig, ax = plt.subplots(figsize=(10, 2))  # Adjust size as needed
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    plt.title(title)
    pdf_pages.savefig(fig)
    plt.close()

# Create a PdfPages object
with PdfPages('combined_output.pdf') as pdf_pages:
    # Core parameters
    core_params = {
        'Parameter': ['dt', 'T', 'dt_ems', 'T_ems', 'T0', 'Ppv_nom'],
        'Value': [dt, T, dt_ems, T_ems, T0, Ppv_nom]
    }
    core_params_df = pd.DataFrame(core_params)
    add_df_to_pdf(core_params_df, "Core Parameters", pdf_pages)

    # EV parameters
    ev_params = {
        'Parameter': ['N_EVs', 'Emax_EV', 'Emin_EV', 'P_max_EV', 'P_min_EV'],
        'Value': [N_EVs, Emax_EV, Emin_EV, P_max_EV, P_min_EV]
    }
    ev_params_df = pd.DataFrame(ev_params)
    add_df_to_pdf(ev_params_df, "EV Parameters", pdf_pages)

    # EV initial conditions
    ev_initial_conditions = {
        'EV Index': list(range(N_EVs)),
        'Initial Energy (E0_EVs)': E0_EVs,
        'Arrival Time (ta_EVs)': ta_EVs,
        'Departure Time (td_EVs)': td_EVs
    }
    ev_initial_conditions_df = pd.DataFrame(ev_initial_conditions)
    add_df_to_pdf(ev_initial_conditions_df, "EV Initial Conditions", pdf_pages)
################
  




