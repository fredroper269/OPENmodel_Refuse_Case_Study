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
#plt.close('all')

############## VERSION ##############


__version__ = "0.0.1"
        
#######################################
###       
### Case Study: Refuse V2G
###        
#######################################

path_string = normpath('Results/RefuseV2G/')
if not os.path.isdir(path_string):
    os.makedirs(path_string)


#######################################
### STEP 0: Load raw Data
#######################################


PV_data_path = os.path.join("Data/Building/", "PVpu_30min_2013JUN.csv") #change this to 30 min data  
PVpu_raw_smr = pd.read_csv(PV_data_path, index_col=0, parse_dates=True).values
Loads_data_path = os.path.join("Data/Building/", "Loads_30min_2013JUN.csv") #THIS HAS BEEN SET TO 0##
Loads_raw_smr = pd.read_csv(Loads_data_path, index_col=0, parse_dates=True).values

PVtotal_smr = np.sum(PVpu_raw_smr,1)


PVpu = PVtotal_smr/np.max(PVtotal_smr)
Loads = Loads_raw_smr

#######################################
### STEP 1: setup parameters
#######################################

dt = 30/60 #30 minute time intervals #was 1/60 #Could change it to 60 for hourly data
T = int(24/dt) #Number of intervals
dt_ems = 15/60 #30 minute EMS time intervals #models this at 1 min #
T_ems = int(T*dt/dt_ems) #Number of EMS intervals
T0 = 0 #from 8 am to 8 am
Ppv_nom = 400 #power rating of the PV generation
#Electric Vehicle (EV) parameters
eff_EV=0.99 #would need to assume no losses in terms of charging and sicharging in method
eff_EV_opt=1
N_EVs = 10 #number of EVs
Emax_EV = 200  #maximum EV energy level #change this to 200 (capacity of eRCV battery)
Emin_EV = 0 #minimum EV energy level #Minimum is 20
P_max_EV = 7 #maximum EV charging power #Fast charger capacity (50kw likely)
P_min_EV = 0 #minimum EV charging power
np.random.seed(1000)
E0_EVs = Emax_EV*np.random.uniform(0,1,N_EVs) #random EV initial energy levels #May want to set that to a single level, at an array so need to set to 1s, they all start on the same amount
ta_EVs = np.random.randint(12*2,22*2,N_EVs) - T0*2 #random EV arrival times between 12pm and 10pm, 
td_EVs = np.random.randint(29*2,32*2+1,N_EVs) - T0*2 #random EV departure times 4am and 8am
#Ensure EVs can be feasibility charged
for i in range(N_EVs):
    td_EVs[i] = np.max([td_EVs[i],ta_EVs[i]])
    E0_EVs[i] = np.max([E0_EVs[i],Emax_EV-P_max_EV*(td_EVs[i]-ta_EVs[i])])

#Market parameters
dt_market = dt_ems #market and EMS have the same time-series
T_market = T_ems #market and EMS have same length
prices_export = 500*np.ones(T_market) #money received of net exports #prices_import = np.array(7,7,7,7,7,15,15) eg.
prices_import = np.hstack((0.07*np.ones(int(T_market*7/24)), \
                          0.15*np.ones(int(T_market*17/24)))) #price of net imports
demand_charge = 0.10 #price per kW for the maximum demand
Pmax_market = 500*np.ones(T_market) #maximum import power ##Would need to change this to make 100 EVs##
Pmin_market = -500*np.ones(T_market) #maximum export power

#######################################
### STEP 2: setup the network
#######################################

#(from https://github.com/e2nIEE/pandapower/blob/master/tutorials/minimal_example.ipynb)
network = pp.create_empty_network()
#create buses 
bus1 = pp.create_bus(network, vn_kv=20., name="bus 1")
bus2 = pp.create_bus(network, vn_kv=0.4, name="bus 2")
bus3 = pp.create_bus(network, vn_kv=0.4, name="bus 3")
#create bus elements
pp.create_ext_grid(network, bus=bus1, vm_pu=1.0, name="Grid Connection")
#create branch elements
trafo = pp.create_transformer(network, hv_bus=bus1, lv_bus=bus2, std_type="0.4 MVA 20/0.4 kV", name="Trafo")
line = pp.create_line(network, from_bus=bus2, to_bus=bus3, length_km=0.1, std_type="NAYY 4x50 SE", name="Line")
N_buses = network.bus['name'].size


#######################################
### STEP 3: setup the assets 
#######################################

#initiate empty lists for different types of assets
storage_assets = []
nondispatch_assets = []

#PV source at bus 3
Pnet = -PVpu*Ppv_nom #100kW PV plant
Qnet = np.zeros(T)
PV_gen_bus3 = AS.NondispatchableAsset(Pnet, Qnet, bus3, dt, T)
nondispatch_assets.append(PV_gen_bus3)

#Load at bus 3
Pnet = np.zeros(T) ##SET LOAD TO 0##
Qnet = np.zeros(T)
load_bus3 = AS.NondispatchableAsset(Pnet, Qnet, bus3, dt, T)
nondispatch_assets.append(load_bus3)

# EVs at bus 634
for i in range(N_EVs): 
    Emax_ev_i = Emax_EV*np.ones(T_ems)
    Emin_ev_i = Emin_EV*np.ones(T_ems)
    Pmax_ev_i = np.zeros(T_ems)
    Pmin_ev_i = np.zeros(T_ems)
    for t in range(ta_EVs[i],int(min(td_EVs[i],T_ems))):
        Pmax_ev_i[t] = P_max_EV
        Pmin_ev_i[t] = P_min_EV
    bus_id_ev_i = bus3
    ev_i = AS.StorageAsset(Emax_ev_i, Emin_ev_i, Pmax_ev_i, Pmin_ev_i,
                           E0_EVs[i], Emax_EV, bus_id_ev_i, dt, T, dt_ems,
                           T_ems, Pmax_abs=P_max_EV, c_deg_lin = 0,
                           eff = eff_EV, eff_opt = eff_EV_opt)
    storage_assets.append(ev_i)

#######################################
### STEP 4: setup the market
#######################################
    
bus_id_market = bus1
market = MK.Market(bus_id_market, prices_export, prices_import, demand_charge, Pmax_market, Pmin_market, dt_market, T_market)

#######################################
#STEP 5: setup the energy system
#######################################

energy_system = ES.EnergySystem(storage_assets, nondispatch_assets, network, market, dt, T, dt_ems, T_ems)

#######################################
### STEP 6: simulate the energy system: 
#######################################

output = energy_system.simulate_network()
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

P_demand_base = np.zeros(T)
for i in range(len(nondispatch_assets)):
    bus_id = nondispatch_assets[i].bus_id
    P_demand_base += nondispatch_assets[i].Pnet
    
#######################################
### STEP 7: plot results
#######################################

#x-axis time values
time = dt*np.arange(T)
time_ems = dt_ems*np.arange(T_ems)
timeE = dt*np.arange(T+1)

#Print revenue generated
revenue = market.calculate_revenue(-Pnet_market,dt)
print('Net Revenue: £ ' + str(revenue) )

#Plot the base demand and the total imported power
plt.figure(num=None, figsize=(6, 3), dpi=80, facecolor='w', edgecolor='k')
plt.plot(time,P_demand_base,'--',label='Base Demand')
plt.plot(time,Pnet_market,label='Total Power Imported')
plt.ylabel('Power (kW)')
plt.xlabel('Time (h)')
plt.xlim(0, max(time))
plt.legend()
plt.grid(True,alpha=0.5)
plt.tight_layout()