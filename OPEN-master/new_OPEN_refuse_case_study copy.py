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

path_string = normpath('Results/RefuseV2G/')
if not os.path.isdir(path_string):
    os.makedirs(path_string)


#######################################
# STEP 0: Load raw Data
#######################################


PV_data_path = os.path.join("/Users/fredroper/Desktop/OPEN_model/OPEN-master", "PV_1minADL.csv")
PVpu_raw_smr = pd.read_csv(PV_data_path, index_col=0, parse_dates=True).values
Loads_data_path = os.path.join("/Users/fredroper/Desktop/OPEN_model/OPEN-master", "Loads_1min.csv")
Loads_raw_smr = pd.read_csv(Loads_data_path, index_col=0, parse_dates=True).values


#######################################
# STEP 1: setup parameters
#######################################

dt = 5/ 60  #5, 30, or 60
T = int(24 / dt)  # Number of intervals
dt_ems = 10 / 60  # 30 or 60.
T_ems = int(T * dt / dt_ems)  # Number of EMS intervals
T0 = 0  # start time of the model relative to input pv and load data e.g. T0=0  is midnight to midnight.
Ppv_nom = 400  # power rating of the PV generation
# Electric Vehicle (EV) parameters
eff_EV = 0.99
eff_EV_opt = 1
N_EVs = 25  # number of EVs
Emax_EV = 30  # maximum EV energy level
Emin_EV = 0  # minimum EV energy level
P_max_EV = 50  # maximum EV charging power
P_min_EV = 0  # minimum EV charging power
np.random.seed(1000)
# EV arrival & departure times and energy levels on arrival
np.random.seed(1000)
# random EV initial energy levels
E0_EVs = Emax_EV*np.random.uniform(0.2,0.9,N_EVs)
# random EV arrival times between 12pm and 1pm
ta_EVs = np.random.randint(int(12/dt_ems),int(13/dt_ems),N_EVs) - int(T0/dt_ems)
# random EV departure times between 3am and 4am
td_EVs = np.random.randint(int(3/dt_ems),\
                           int(4/dt_ems),N_EVs) - int(T0/dt_ems)

# ta_EVs = np.ones(N_EVs) * 12
# td_EVs = np.ones(N_EVs) * 12

# Ensure EVs can be feasibility charged
for i in range(N_EVs):
    td_EVs[i] = np.max([td_EVs[i], ta_EVs[i]])
    E0_EVs[i] = np.max([E0_EVs[i], Emax_EV - P_max_EV * (td_EVs[i] - ta_EVs[i])])

# Market parameters
dt_market = dt_ems  # market and EMS have the same time-series #Need to find data on Aus electricity prices#
T_market = T_ems  # market and EMS have same length
prices_export = 0.85 * np.ones(T_market)  # money received of net exports
prices_import = np.hstack((0.70 * np.ones(int(T_market * 7 / 24)),
                          0.02 * np.ones(int(T_market * 2 / 24)),
                          0.80 * np.ones(int(T_market * 15 / 24))
                           ))  # price of net imports

demand_charge = 0.10  # price per kW for the maximum demand
Pmax_market = 5000 * np.ones(T_market)  # maximum import power
Pmin_market = -5000 * np.ones(T_market)  # maximum export power

#######################################
# STEP 1b: wrangle input data
#######################################
# convert to same time resolution as defined by dt
PVpu_conv = np.array([np.interp(np.arange(0, len(PVpu_raw_smr[:, pv]), PVpu_raw_smr.shape[0] / T), np.arange(0, len(PVpu_raw_smr[:, pv])), PVpu_raw_smr[:, pv]) for pv in range(PVpu_raw_smr.shape[1])]).T
Loads_conv = np.array([np.interp(np.arange(0, len(Loads_raw_smr[:, l]), Loads_raw_smr.shape[0] / T), np.arange(0, len(Loads_raw_smr[:, l])), Loads_raw_smr[:, l]) for l in range(Loads_raw_smr.shape[1])]).T


PVtotal_smr = np.sum(PVpu_conv, 1)
PVpu = PVtotal_smr #/ np.max(PVtotal_smr)
Loads = Loads_conv

# if you want zero load, uncomment line below
Loads = np.zeros(Loads.shape)

# if you want zero PV, uncomment line below
#PVpu = np.zeros(PVpu.shape)


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
trafo = pp.create_transformer(network, hv_bus=bus1, lv_bus=bus2, std_type="0.4 MVA 20/0.4 kV", name="Trafo")
line = pp.create_line(network, from_bus=bus2, to_bus=bus3, length_km=0.1, std_type="NAYY 4x50 SE", name="Line")
N_buses = network.bus['name'].size


#######################################
# STEP 3: setup the assets
#######################################

# initiate empty lists for different types of assets
storage_assets = []
nondispatch_assets = []

# PV source at bus 3
Pnet = -PVpu * Ppv_nom  # 100kW PV plant
Qnet = np.zeros(T)
PV_gen_bus3 = AS.NondispatchableAsset(Pnet, Qnet, bus3, dt, T)
nondispatch_assets.append(PV_gen_bus3)

# Load at bus 3
Pnet = np.sum(Loads, 1)  # summed load across 120 households
Qnet = np.zeros(T)
load_bus3 = AS.NondispatchableAsset(Pnet, Qnet, bus3, dt, T)
nondispatch_assets.append(load_bus3)

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
#output = energy_system.simulate_network_bldg()

buses_Vpu = output['buses_Vpu']
buses_Vang = output['buses_Vang']
buses_Pnet = output['buses_Pnet']
buses_Qnet = output['buses_Qnet']
Pnet_market = output['Pnet_market']
Qnet_market = output['Qnet_market']
buses_Vpu = output['buses_Vpu']
P_import_ems = output['P_import_ems']
P_export_ems = output['P_export_ems']
# P_BLDG_ems = output['P_BLDG_ems']
P_demand_ems = output['P_demand_ems']
P_ES_ems = output['P_ES_ems']


P_demand_base = np.zeros(T)
for i in range(len(nondispatch_assets)):
    bus_id = nondispatch_assets[i].bus_id
    P_demand_base += nondispatch_assets[i].Pnet

#######################################
# STEP 7: plot results
#######################################

# x-axis time values
time = dt * np.arange(T)
time_ems = dt_ems * np.arange(T_ems)
timeE = dt * np.arange(T + 1)

# Print revenue generated
revenue = market.calculate_revenue(Pnet_market, dt)
print('Net Revenue: £ ' + str(revenue))

# Plot the base demand and the total imported power
plt.figure(num=None, figsize=(6, 3), dpi=80, facecolor='w', edgecolor='k')
plt.plot(time, P_demand_base, '--', label='Base Demand')
plt.plot(time, Pnet_market, label='Total Power Imported')
plt.ylabel('Power (kW)')
plt.xlabel('Time (h)')
plt.xlim(0, max(time))
plt.legend()
plt.grid(True, alpha=0.5)
plt.tight_layout()


# PF_network_res = output['PF_network_res']
P_import_ems = output['P_import_ems']
P_export_ems = output['P_export_ems']
P_ES_ems = output['P_ES_ems']
P_demand_ems = output['P_demand_ems']

P_demand_base = np.zeros(T)
for i in range(len(nondispatch_assets)):
    bus_id = nondispatch_assets[i].bus_id
    P_demand_base += nondispatch_assets[i].Pnet

P_demand_base_pred = np.zeros(T)
for i in range(len(nondispatch_assets)):
    bus_id = nondispatch_assets[i].bus_id
    P_demand_base_pred += nondispatch_assets[i].Pnet_pred

# Pnet_market = np.zeros(T)
# for t in range(T):
#     market_bus_res = PF_network_res[t].res_bus_df.iloc[bus_id_market]
#     Pnet_market[t] = np.real\
#                     (market_bus_res['Sa']\
#                      + market_bus_res['Sb']\
#                      + market_bus_res['Sc'])

# N_phases = network.N_phases  # Number of phases
N_ESs = len(storage_assets)  # number of storage assets

# buses_Vpu = np.zeros([T,N_buses,N_phases])
# for t in range(T):
#     for bus_id in range(N_buses):
#         bus_res = PF_network_res[t].res_bus_df.iloc[bus_id]
#         buses_Vpu[t,bus_id,0] = np.abs(bus_res['Va'])/network.Vslack_ph
#         buses_Vpu[t,bus_id,1] = np.abs(bus_res['Vb'])/network.Vslack_ph
#         buses_Vpu[t,bus_id,2] = np.abs(bus_res['Vc'])/network.Vslack_ph

P_demand_base_pred_ems = np.zeros(T_ems)
for t_ems in range(T_ems):
    t_indexes = (t_ems * dt_ems / dt + np.arange(0, dt_ems / dt)).astype(int)
    P_demand_base_pred_ems[t_ems] = np.mean(P_demand_base_pred[t_indexes])

EVs_tot = sum(P_ES_ems[:, n] for n in range(N_ESs))
P_compare = P_demand_base_pred_ems + EVs_tot
#######################################
# STEP 7: plot results
#######################################

# x-axis time values
time = dt * np.arange(T)
time_ems = dt_ems * np.arange(T_ems)
timeE = dt * np.arange(T + 1)

# energy cost
energy_cost = market.calculate_revenue(Pnet_market, dt)
energy_cost_string = 'Total energy cost: £ %.2f' % (-1 * energy_cost)
print(energy_cost_string)

# #save the data
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


def figure_plot(x, N_EVs, P_demand_base_pred_ems, P_compare, P_demand_base,
                Pnet_market, storage_assets, N_ESs,
                nondispatch_assets, time_ems, time, timeE, buses_Vpu, save_suffix='.pdf'):

    # plot half hour predicted and actual net load
    title = str(x)
    # plt.figure(num=None, figsize=(6, 2.5), dpi=80, facecolor='w', edgecolor='k')
    # plt.title(label="predicted vs actual")
    # plt.plot(time_ems,P_demand_base_pred_ems,label=\
    #          'Predicted net load, 30 mins')
    # plt.plot(time_ems,P_compare, label =\
    #          'Predicted net load + EVs charging, 30 mins')
    # plt.ylabel('Power (kW)')
    # # plt.ylim(0, 2100)
    # plt.xticks([0,8,16,23.75],('00:00', '08:00', '16:00', '00:00'))
    # plt.xlabel('Time (hh:mm)')
    # plt.xlim(0, max(time_ems))
    # plt.grid(True,alpha=0.5)
    # # plt.legend()
    # plt.tight_layout()
    # ax = plt.gca()
    # plt.text(0.02, 0.9, title, transform=ax.transAxes, fontsize=12)
    # plt.savefig(join(path_string, normpath('P_ems_'  + str(x) + save_suffix)),
    #             bbox_inches='tight')

    # plot 5 minute predicted and actual net load
    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=80, facecolor='w', edgecolor='k')
    ax.set_title(label="base net load vs actual net load")
    ax.step(time, P_demand_base, '--', label='Base Load', where='post')
    ax.step(time, Pnet_market, label='Import Power', where='post')
    ax.set_ylabel('Power (kW)')
    # plt.ylim(500, 2100)
    ax.set_xticks([0, 8, 16, 23.916], ('00:00', '08:00', '16:00', '00:00'))
    ax.set_xlabel('Time (hh:mm)')
    ax.set_xlim(0, max(time))
    
    ax2 = ax.twinx()
    ax2.step(time_ems, prices_import, where='post', linestyle='--', linewidth=0.7, color='k')
    ax2.set_ylabel('Import price (£/kWh)')
    
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    ax = plt.gca()
    plt.text(0.02, 0.9, title, transform=ax.transAxes, fontsize=12)
    plt.savefig(join(path_string, normpath('P_actual_' + str(x) + save_suffix)),
                bbox_inches='tight')

    # plot power for EV charging
    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=80, facecolor='w', edgecolor='k')
    ax.set_title(label="Storage asset power")
    #plt.plot(time,sum(storage_assets[i].Pnet for i in range(N_ESs)))
    for i in range(N_EVs):
        ax.step(time, storage_assets[i].Pnet, where='post')
    # ax.set_xlim(0, 24)
    # plt.ylim(0, 10)
    ax2 = ax.twinx()
    ax2.step(time_ems, prices_import, where='post', linestyle='--', linewidth=0.7, color='k')
    ax2.set_ylabel('Import price (£/kWh)')
    ax.set_ylabel('Power (kW)')
    ax.set_xticks([0, 8, 16, 23.916], ('00:00', '08:00', '16:00', '00:00'))
    ax.set_xlabel('Time (hh:mm)')
    ax.set_xlim(0, max(time))
    plt.grid(True, alpha=0.5)
    ax = plt.gca()
    plt.tight_layout()
    plt.text(0.02, 0.9, title, transform=ax.transAxes, fontsize=12)
    plt.savefig(join(path_string, normpath('P_EVs_' + str(x) + save_suffix)),
                bbox_inches='tight')

    # plot average battery energy
    plt.figure(num=None, figsize=(6, 2.5), dpi=80, facecolor='w', edgecolor='k')
    plt.title(label="average battery energy")
    plt.step(timeE, sum(storage_assets[i].E for i in range(N_ESs)) / N_EVs, where='post')
    plt.ylabel('Average EV Energy (kWh)')
    plt.xticks([0, 8, 16, 23.916], ('00:00', '08:00', '16:00', '00:00'))
    plt.yticks(np.arange(0, 37, 4))
    plt.ylim(12, 36)
    plt.xlabel('Time (hh:mm)')
    plt.xlim(0, max(time))
    plt.grid(True, alpha=0.5)
    ax = plt.gca()
    plt.tight_layout()
    plt.text(0.02, 0.9, title, transform=ax.transAxes, fontsize=12)
    plt.savefig(join(path_string, normpath('E_EVs_' + str(x) + save_suffix)),
                bbox_inches='tight')


figure_plot('Refuse Case Study', N_EVs, P_demand_base_pred_ems, P_compare, P_demand_base,
            Pnet_market, storage_assets, N_ESs,
            nondispatch_assets, time_ems, time, timeE, buses_Vpu)
