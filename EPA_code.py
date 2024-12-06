# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 09:55:02 2024

@author: Cassidy Hartog and Nizam Uddin
"""
### 11/07/2024

import numpy as np
import pandas as pd
from pandas import isna
import lhs
import math
import scipy
from scipy import stats
from setup import setup_data
import copy
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick



#%% LCC/LCA modeling inputs

# # outputs of interest
output_perc_mid = 50
output_perc_low = 5
output_perc_high = 95

# general parameters - import spreadsheet tabs as dataframes
design_assumptions = pd.read_excel('Assumptions_AD.xlsx', sheet_name = 'Design', index_col='Parameter')


# number of Monte Carlo runs
n_samples = int(design_assumptions.loc['n_samples','expected'])

expected_lifetime = int(design_assumptions.loc['expected_lifetime','expected'])


# create empty datasets to eventually store data for sensitivity analysis (Spearman's coefficients)
correlation_distributions = np.full((n_samples, n_samples), np.nan)
correlation_parameters = np.full((n_samples, 1), np.nan)
correlation_parameters = correlation_parameters.tolist()


# loading in all of the data
result = setup_data([design_assumptions],correlation_distributions,correlation_parameters,n_samples)



# creates variables for each of the variables in the excel file
for key in result.keys():
    exec(f'{key} = result[key]')


#%% run design 

#inputs

#foodwaste (FW) characterization 
FW_influent_volume = Amount_foodwaste / density_foodwaste #m3 / d total solids and moisture content
FW_influent_solids = Amount_foodwaste * (1 - Moisture_content_FW) # kg total solids / d
FW_influent_COD = COD_foodwaste * (1 / density_foodwaste) * FW_influent_solids / 1000 # Kg COD / d


#pulp and paper mill sludge (PPMS) characterization
PPMS_influent_volume = (Amount_PPMS / Density_PPMS) / 1000 # m3 / d total with solids and moisture
PPMS_influent_solids = Amount_PPMS * (1 - Moisture_content_PPMS)  # kg total solids / d
PPMS_influent_COD = COD_ppmill_sludge * PPMS_influent_solids * (1 / Density_PPMS) * 1e-6 # kg COD / d
PPMS_influent_VSS = VS_PPMS * PPMS_influent_volume # kg VS / d

#Region settings 

region = region[0][0]       
if region == 1:
    Tipping_fees_US = Tipping_fees_US_1
    methane_generation_constant = k_high_rain
    region_energy_GWP = coal_use_US_1 * coal_GHG + oil_use_US_1 * oil_GHG + gas_use_US_1 * gas_GHG + hydro_use_US_1 * hydro_GHG + bio_use_US_1 * bio_GHG + geo_use_US_1 * geo_GHG + wind_use_US_1 * wind_GHG + solar_use_US_1 * solar_GHG + nuclear_use_US_1 * nuclear_GHG # kg CO2 eq / Kwh
    Electricity_cost = Electricity_cost_1
    T = 10
    transport_distance=Transport_distance_US_1
elif region == 2:
    Tipping_fees_US = Tipping_fees_US_2
    methane_generation_constant = k_high_rain
    region_energy_GWP = coal_use_US_2 * coal_GHG + oil_use_US_2 * oil_GHG + gas_use_US_2 * gas_GHG + hydro_use_US_2 * hydro_GHG + bio_use_US_2 * bio_GHG + geo_use_US_2 * geo_GHG + wind_use_US_2 * wind_GHG + solar_use_US_2 * solar_GHG + nuclear_use_US_2 * nuclear_GHG # kg CO2 eq / Kwh
    Electricity_cost = Electricity_cost_2
    T = 15
    transport_distance=Transport_distance_US_2
elif region == 3:
    Tipping_fees_US = Tipping_fees_US_3
    methane_generation_constant = k_high_rain
    region_energy_GWP = coal_use_US_3 * coal_GHG + oil_use_US_3 * oil_GHG + gas_use_US_3 * gas_GHG + hydro_use_US_3 * hydro_GHG + bio_use_US_3 * bio_GHG + geo_use_US_3 * geo_GHG + wind_use_US_3 * wind_GHG + solar_use_US_3 * solar_GHG + nuclear_use_US_3 * nuclear_GHG # kg CO2 eq / Kwh
    Electricity_cost = Electricity_cost_3
    T = 10
    transport_distance=Transport_distance_US_3
elif region == 4:
    Tipping_fees_US = Tipping_fees_US_4
    methane_generation_constant = k_low_rain
    region_energy_GWP = coal_use_US_4 * coal_GHG + oil_use_US_4 * oil_GHG + gas_use_US_4 * gas_GHG + hydro_use_US_4 * hydro_GHG + bio_use_US_4 * bio_GHG + geo_use_US_4 * geo_GHG + wind_use_US_4 * wind_GHG + solar_use_US_4 * solar_GHG + nuclear_use_US_4 * nuclear_GHG # kg CO2 eq / Kwh
    Electricity_cost = Electricity_cost_4
    T = 15
    transport_distance=Transport_distance_US_4
elif region == 5:
    Tipping_fees_US = Tipping_fees_US_5
    methane_generation_constant = k_low_rain
    region_energy_GWP = coal_use_US_5 * coal_GHG + oil_use_US_5 * oil_GHG + gas_use_US_5 * gas_GHG + hydro_use_US_5 * hydro_GHG + bio_use_US_5 * bio_GHG + geo_use_US_5 * geo_GHG + wind_use_US_5 * wind_GHG + solar_use_US_5 * solar_GHG + nuclear_use_US_5 * nuclear_GHG # kg CO2 eq / Kwh
    Electricity_cost = Electricity_cost_5
    T = 10
    transport_distance=Transport_distance_US_5
elif region == 6:
    Tipping_fees_US = Tipping_fees_US_6
    methane_generation_constant = k_low_rain
    region_energy_GWP = coal_use_US_6 * coal_GHG + oil_use_US_6 * oil_GHG + gas_use_US_6 * gas_GHG + hydro_use_US_6 * hydro_GHG + bio_use_US_6 * bio_GHG + geo_use_US_6 * geo_GHG + wind_use_US_6 * wind_GHG + solar_use_US_6 * solar_GHG + nuclear_use_US_6 * nuclear_GHG # kg CO2 eq / Kwh
    Electricity_cost = Electricity_cost_6
    T = 10
    transport_distance=Transport_distance_US_6
elif region == 7:
    Tipping_fees_US = Tipping_fees_US_7
    methane_generation_constant = (k_low_rain + k_high_rain) / 2
    region_energy_GWP = coal_use_US_7 * coal_GHG + oil_use_US_7 * oil_GHG + gas_use_US_7 * gas_GHG + hydro_use_US_7 * hydro_GHG + bio_use_US_7 * bio_GHG + geo_use_US_7 * geo_GHG + wind_use_US_7 * wind_GHG + solar_use_US_7 * solar_GHG + nuclear_use_US_7 * nuclear_GHG # kg CO2 eq / Kwh
    Electricity_cost = Electricity_cost_7
    T = 20
    transport_distance=Transport_distance_US_7


#%%FW GHG emissions


#transport GHG for FW
FW_landfill_total_mass = Amount_foodwaste / 1000 # ton of FW / d
FW_landfill_num_trips = np.ceil(FW_landfill_total_mass / truck_capacity) * 2 # num of two-way trips to landfill / d
transport_CO2_emissions_FW = Landfill_distance_US * Transport_emissions_factor_truck *FW_landfill_total_mass* FW_landfill_num_trips # kg CO2eq / d


FW_methane_gen_potential = 109 #m3 CH4 / Mg waste using conversion factors from LANDGEM EPA ##Find a value
FW_refuse_acceptance_rt = FW_influent_solids * 365/1000 # Mg refuse fw / yr
FW_landfill_methane_gen_rt = []

for index, i in enumerate(Time_since_landfill_placement): 
    
    output= 1.3 * FW_methane_gen_potential * int(FW_refuse_acceptance_rt[index]) * (math.exp(np.percentile(-methane_generation_constant, output_perc_mid) * np.percentile(Time_since_landfill_close, output_perc_mid)) - math.exp(np.percentile(-methane_generation_constant, output_perc_mid) * i)) # m3 CH4 / yr
    FW_landfill_methane_gen_rt.append([output])

FW_landfill_methane_gen_rt_array = FW_landfill_methane_gen_rt * dummy

FW_landfill_methane_released = FW_landfill_methane_gen_rt_array * 0.25 # m3 CH4 released into atmosphere / yr, assuming 75% is captured and burned (from Methane EPA 2008)

#%%Cost FW

#transport cost for FW
FW_Tipping_fee = (Tipping_fees_US * Amount_foodwaste / 1000) * 365 #USD / yr
FW_transport_cost =  ((((Landfill_distance_US / Average_speed) * Driver_wages_US  * FW_landfill_num_trips) 
                       + (Landfill_distance_US * diesel_transport * FW_landfill_total_mass * FW_landfill_num_trips * diesel_cost_US))  * 365) #USD / yr 



#%%anaerobic digestor (AD) PPMS
if T == 10:
    T_wall = copy.deepcopy(T_wall_10)
    T_floor = copy.deepcopy(T_floor_10)
    T_roof = copy.deepcopy(T_roof_10)
elif T == 15:
    T_wall = copy.deepcopy(T_wall_15)
    T_floor = copy.deepcopy(T_floor_15)
    T_roof = copy.deepcopy(T_roof_15)
elif T == 20:
    T_wall = copy.deepcopy(T_wall_20)
    T_floor = copy.deepcopy(T_floor_20)
    T_roof = copy.deepcopy(T_roof_20)
elif T == 25:
    T_wall = copy.deepcopy(T_wall_25)
    T_floor = copy.deepcopy(T_floor_25)
    T_roof = copy.deepcopy(T_roof_25)

Q = PPMS_influent_volume
N_AD = 1        
wall_depth_AD = AD_depth + AD_freeboard 
floor_angle = math.atan(np.percentile(AD_bottom_slope, output_perc_mid))
heat_sludge = (Cp_sludge/1000) * (Q * 1000) * (Mesophilic_AD_temperature - T)  # MJ/d
heat_wall = (U_wall * 3600 * 24 / 10**6) * (np.pi * Reactor_diameter * wall_depth_AD * N_AD) * (Mesophilic_AD_temperature - T_wall)
heat_floor = (U_floor * 3600 * 24 / 10**6) * (N_AD * np.pi/4*(Reactor_diameter/math.cos(floor_angle))**2) * (Mesophilic_AD_temperature - T_floor)
heat_roof = (U_roof * 3600 * 24 / 10**6) * (N_AD * np.pi/4*(Reactor_diameter)**2) * (Mesophilic_AD_temperature - T_roof)
heat_required_AD = heat_sludge + heat_wall + heat_floor + heat_roof     # MJ/d


#AD methane generation

total_VS_PPMS = VS_PPMS * PPMS_influent_volume  # kg/day 
methane_production_PPMS = total_VS_PPMS * Specific_methane_yield_PPMS   # m3 / d CH4
methane_energy = methane_production_PPMS * Energy_content_methane  # MJ/d
methane_energy_converted = methane_energy * 0.95 # MJ/d of gas that does not escape
methane_heat_generation = methane_energy_converted * CHP_heat_efficiency  # MJ/d
AD_heat_balance = methane_heat_generation - heat_required_AD    # MJ/d excess heat
AD_excess_energy_GWP = -AD_heat_balance * (1/3.6) * region_energy_GWP #kg CO2 eq / d  (1 Kw-h = 3.6 MJ)

# PPMS after digestion
VS_destroyed = AD_VS_destruction * total_VS_PPMS  # kg VS / d destroyed
PPMS_post_AD_solids = PPMS_influent_solids - VS_destroyed  # kg/d 
PPMS_post_AD_COD = PPMS_influent_COD - (PPMS_influent_COD * COD_degredation) # kg COD / d

#%%PPMS GHG emissions

#transport GHG
PPMS_landfill_total_mass = PPMS_post_AD_solids / 1000 # ton of ppms / d
PPMS_landfill_num_trips = np.ceil(PPMS_landfill_total_mass / truck_capacity) * 2 # num of two-way trips to landfill / d
transport_CO2_emissions_PPMS = Landfill_distance_US * Transport_emissions_factor_truck * PPMS_landfill_total_mass * PPMS_landfill_num_trips # kg CO2eq / d


#landfill GHG PPMS
PPMS_methane_gen_potential = (PPMS_post_AD_COD * 0.35 )/(PPMS_post_AD_solids/1000) #m3 CH4 / Mg waste using conversion factors (from LANDGEM EPA)
PPMS_refuse_acceptance_rt = PPMS_post_AD_solids * 365/1000 # Mg refuse ppms / yr



PPMS_landfill_methane_gen_rt = 1.3 * PPMS_methane_gen_potential * PPMS_refuse_acceptance_rt * (math.exp(np.percentile(-methane_generation_constant, output_perc_mid) * np.percentile(Time_since_landfill_close, output_perc_mid)) - math.exp(np.percentile(-methane_generation_constant, output_perc_mid) * np.percentile(Time_since_landfill_placement, output_perc_mid))) # m3 CH4 / yr
PPMS_landfill_methane_released = PPMS_landfill_methane_gen_rt * 0.25 # m3 CH4 released into atmosphere / yr, assuming 75% is captured and burned (from Methane EPA 2008)


#%%Cost PPMS 

#AD operational cost
PPMS_AD_heating_cost = (-AD_heat_balance / 3.6) * 365 * Electricity_cost #USD / yr

#transport to landfill
PPMS_Tipping_fee = (Tipping_fees_US * PPMS_post_AD_solids / 1000) * 365 #USD/yr
PPMS_transport_cost =  ((((Landfill_distance_US / Average_speed) * Driver_wages_US  * PPMS_landfill_num_trips) 
                       + (Landfill_distance_US * diesel_transport * PPMS_landfill_total_mass * PPMS_landfill_num_trips * diesel_cost_US)) * 365) #USD / yr


#%%integrated FW and PPMS
#characterize mixed FW and sludge
total_VS_mixed = VS_PPMS * PPMS_influent_volume + FW_VS* FW_influent_volume  # kg VSS / d
total_influent_volume = FW_influent_volume + PPMS_influent_volume #m3/d
total_influent_solids = FW_influent_solids + PPMS_influent_solids #kg total solids/d
total_influent_COD = FW_influent_COD + PPMS_influent_COD # kg COD / d
total_influent_VSS = total_VS_mixed  # kg VSS / d
mixed_COD_VSS_content = total_influent_COD / total_influent_VSS # g COD / g VSS

#AD methane generation mixed

methane_production_mixed = total_VS_mixed * Specific_methane_yield_mixed # m3/d CH4
methane_energy_mixed = methane_production_mixed * Energy_content_methane  # MJ/d
methane_energy_converted_mixed = methane_energy_mixed * 0.95 # MJ/d of gas that does not escape
methane_heat_generation_mixed = methane_energy_converted_mixed * CHP_heat_efficiency  # MJ/d
AD_heat_balance_mixed = methane_heat_generation_mixed - heat_required_AD    # MJ/d excess heat
AD_excess_energy_GWP_mixed = -AD_heat_balance_mixed * 1/3.6 * region_energy_GWP #kg CO2 eq / d

#mixed FW and sludge after AD
VS_destroyed_mixed = AD_VS_destruction_mixed * total_VS_mixed  # kg VS / d destroyed
mixed_post_AD_solids = total_influent_solids - VS_destroyed_mixed # kg/d
mixed_post_AD_COD = total_influent_COD - (total_influent_COD * COD_degredation_mixed) # kg COD / d

#%%mixed emissions

#transport FW to ppmill AD GHG
transport_CO2_emissions_FW_AD = transport_distance * Transport_emissions_factor_truck * FW_landfill_total_mass * FW_landfill_num_trips # kg CO2eq / d

#transport mixed GHG
mixed_landfill_total_mass = mixed_post_AD_solids / 1000 # ton of FW / d
mixed_landfill_num_trips = np.ceil(mixed_landfill_total_mass / truck_capacity) * 2 # num of two-way trips to landfill / d
transport_CO2_emissions_mixed = Landfill_distance_US * Transport_emissions_factor_truck * mixed_landfill_total_mass * mixed_landfill_num_trips # kg CO2eq / d


#landfill mixed GHG
mixed_methane_gen_potential = (mixed_post_AD_COD * 0.35)/(mixed_post_AD_solids/1000) #m3 CH4 / Mg waste using conversion factors from LANDGEM EPA
mixed_refuse_acceptance_rt = mixed_post_AD_solids * 365/1000 # Mg refuse ppms / yr



mixed_landfill_methane_gen_rt = 1.3 * mixed_methane_gen_potential * mixed_refuse_acceptance_rt * (math.exp(np.percentile(-methane_generation_constant, output_perc_mid) * np.percentile(Time_since_landfill_close, output_perc_mid)) - math.exp(np.percentile(-methane_generation_constant, output_perc_mid) * np.percentile(Time_since_landfill_placement, output_perc_mid))) # m3 CH4 / yr
mixed_landfill_methane_released = mixed_landfill_methane_gen_rt * 0.25 # m3 CH4 released into atmosphere / yr, assuming 75% is captured and burned from Methane EPA 2008

# NaOH GHG emissions from mixed system
NaOH_amount_total_mixed = FW_influent_volume * NaOH_amount * 365 # kg/year NaOH addition into the system
NaOH_CO2_emissions_mixed = NaOH_amount_total_mixed * NaOH_GWP * (1/365) # kg CO2 eq/d

#%%integrated FW and PPMS cost

#FW to AD transport cost
FW_to_AD_transport_cost =  ((((transport_distance / Average_speed) * Driver_wages_US  * FW_landfill_num_trips) 
                       + (transport_distance * diesel_transport * FW_landfill_total_mass * FW_landfill_num_trips * diesel_cost_US)) * 365) #USD / yr

#AD operational cost
mixed_AD_heating_cost = (-AD_heat_balance_mixed / 3.6) * 365 * Electricity_cost #USD / yr

#transport to landfill
mixed_Tipping_fee = (Tipping_fees_US * mixed_post_AD_solids / 1000) * 365 #USD / yr
mixed_transport_cost =  ((((Landfill_distance_US / Average_speed) * Driver_wages_US  * mixed_landfill_num_trips) 
                       + (Landfill_distance_US * diesel_transport * mixed_landfill_total_mass * mixed_landfill_num_trips * diesel_cost_US)) * 365) #USD / yr

#NaOH cost for mixed system

NaOH_cost_mixed = (NaOH_amount_total_mixed/1000)*cost_NaOH_USA # USD/year

#%% results

#GHG totals
FW_GHG = (((FW_landfill_methane_released * 0.671 * CH4_GWP) / 365) + transport_CO2_emissions_FW ) / (FW_influent_solids/1000)   # kg CO2 eq / tonne waste/ day
PPMS_GHG = ((methane_production_PPMS * 0.671 * CH4_GWP * 0.05) + AD_excess_energy_GWP + transport_CO2_emissions_PPMS + ((PPMS_landfill_methane_released * 0.671 * CH4_GWP) / 365)) / (PPMS_influent_solids/1000)   # kg CO2 eq / tonne waste
Mixed_GHG = (transport_CO2_emissions_FW_AD + (methane_production_mixed * 0.671 * CH4_GWP * 0.05) + AD_excess_energy_GWP_mixed + transport_CO2_emissions_mixed + ((mixed_landfill_methane_released * 0.671 * CH4_GWP) / 365)+ NaOH_CO2_emissions_mixed) / (total_influent_solids/1000)   # kg CO2 eq / tonne waste

#cost totals
FW_cost = ((FW_Tipping_fee + FW_transport_cost) / (FW_influent_solids / 1000)) / 365 # USD / tonne/ day
PPMS_cost = ((PPMS_AD_heating_cost + PPMS_transport_cost + PPMS_Tipping_fee) / (PPMS_influent_solids / 1000)) / 365 # USD / tonne
Mixed_cost = ((FW_to_AD_transport_cost + mixed_AD_heating_cost + mixed_Tipping_fee + mixed_transport_cost+ NaOH_cost_mixed) / (total_influent_solids / 1000)) / 365 # USD / tonne


titles_GHG = ('FW_GHG', 'PPMS_GHG', 'Mixed_GHG')
# note that there's an index below on FW only GHG
all_GHG = (FW_GHG, PPMS_GHG, Mixed_GHG)

titles_cost = ('FW_cost', 'PPMS_cost', 'Mixed_cost')
all_cost = (FW_cost, PPMS_cost, Mixed_cost)

writer = pd.ExcelWriter('all_results.xlsx', engine='xlsxwriter')

#normalized costs
df_GHG = pd.DataFrame({k:v.flatten() for k,v in zip(titles_GHG, all_GHG)})

df_GHG.to_excel(writer, sheet_name = 'all_GHG')


df_cost = pd.DataFrame({k:v.flatten() for k,v in zip(titles_cost, all_cost)})

df_cost.to_excel(writer, sheet_name = 'all_cost')

#%% sensitivity

all_inputs_name =[]

all_inputs = []

all_inputs = []

for item in result.keys():
    if isinstance(result[item], np.ndarray)==True:
        all_inputs_name.append(item)
        all_inputs.append(result[item])
dfinputs = pd.DataFrame({k:v.flatten() for k,v in zip(all_inputs_name, all_inputs)})

sensitivity_FW_GHG = (dfinputs.corrwith(df_GHG.FW_GHG, method='spearman'))
sensitivity_PPMS_GHG = (dfinputs.corrwith(df_GHG.PPMS_GHG, method='spearman'))
sensitivity_mixed_GHG = (dfinputs.corrwith(df_GHG.Mixed_GHG, method='spearman'))

sensitivity_FW_cost = (dfinputs.corrwith(df_cost.FW_cost, method='spearman'))
sensitivity_PPMS_cost = (dfinputs.corrwith(df_cost.PPMS_cost, method='spearman'))
sensitivity_mixed_cost = (dfinputs.corrwith(df_cost.Mixed_cost, method='spearman'))

dfinputs.to_excel(writer, sheet_name= 'inputs')
sensitivity_FW_GHG.to_excel(writer, sheet_name= 'FW_GHG_spearman')
sensitivity_PPMS_GHG.to_excel(writer, sheet_name= 'PPMS_GHG_spearman')
sensitivity_mixed_GHG.to_excel(writer, sheet_name= 'Mixed_GHG_spearman')

sensitivity_FW_cost.to_excel(writer, sheet_name= 'FW_cost_spearman')
sensitivity_PPMS_cost.to_excel(writer, sheet_name= 'PPMS_cost_spearman')
sensitivity_mixed_cost.to_excel(writer, sheet_name= 'Mixed_cost_spearman')

spearmans_results = pd.DataFrame(dict(

    FW_cost = sensitivity_FW_cost, PPMS_cost = sensitivity_PPMS_cost, Mixed_cost = sensitivity_mixed_cost,
    FW_GHG = sensitivity_FW_GHG, PPMS_GHG = sensitivity_PPMS_GHG, Mixed_GHG = sensitivity_mixed_GHG,
         

    )).reset_index()

spearmans_results.to_excel(writer, sheet_name= 'all spearmans')
X= list(spearmans_results)
X.remove('index')
Y= list(spearmans_results['index'])
spearmans_results_scatter = pd.melt(spearmans_results, id_vars=['index'], value_vars=X)

#%% ouput data to excel file
writer.close()


#%% graph

#set colors
colors = ['#ED586F', '#60c1cf', 
          '#79bf82']


#cost
fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111)
meanprops = {
    "marker": ".",        # Shape of the marker ('o', 's', '^', etc.)
    "markerfacecolor": "black",  # Color of the marker face
    "markeredgecolor": "black", # Color of the marker edge
    "markeredgewidth": 0.5,
    "markersize": 3       # Size of the marker
     }


# Creating axes instance
cost_plot = ax.boxplot(df_cost, patch_artist = True, widths=0.6, flierprops={'markersize': 2}, showfliers= False, whis=[5,95], showmeans= True, meanprops=meanprops)

for patch, color in zip(cost_plot['boxes'], colors):
    patch.set_facecolor(color)
ax.set_xticklabels(['FW', 'PPMS', 'Mixed'])

for median in cost_plot['medians']:
    median.set(color ='black',linewidth = 1)

# Set the font name for axis tick labels to be Comic Sans
for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
for tick in ax.get_yticklabels():
    tick.set_fontname("Arial")
    
# Change the y axis label to Arial
ax.set_ylabel("Cost \n [USD/tonne/day]",fontname='Arial', fontsize=10)
ax.set_ylim([0,1000])

# Change the x axis label to Arial
ax.set_xlabel("Different treatment scenario", fontname="Arial", fontsize=10)

plt.tick_params(direction='inout')
#plt.savefig("Cost_09022024.png", dpi=1000, bbox_inches="tight")    


#GWP

fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111)

# Creating axes instance
cost_plot = ax.boxplot(df_GHG, patch_artist = True, widths=0.6, flierprops={'markersize': 2}, showfliers= False, whis=[5,95], showmeans= True, meanprops=meanprops)

for patch, color in zip(cost_plot['boxes'], colors):
    patch.set_facecolor(color)
ax.set_xticklabels(['FW', 'PPMS', 'Mixed']) 
 

for median in cost_plot['medians']:
    median.set(color ='black',linewidth = 1)

# Set the font name for axis tick labels to be Comic Sans
for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
for tick in ax.get_yticklabels():
    tick.set_fontname("Arial")
    
# Change the y axis label to Arial
ax.set_ylabel('Global warming potential \n [kg CO$_{2}$ eq/tonne/day]', fontname="Arial", fontsize=10)
ax.set_ylim([0,1000])

# Change the x axis label to Arial
ax.set_xlabel("Different treatment scenario", fontname="Arial", fontsize=10)

plt.tick_params(direction='inout')
#plt.savefig("GWP_09022024.png", dpi=1000, bbox_inches="tight")


# spearmans
spearmans_results_scatter['value'] = np.where(abs(spearmans_results_scatter['value'])<0.05, np.nan, spearmans_results_scatter['value'])
spearmans_results_scatter["value"] *= 300
spearmans_results_scatter['value'] = abs(spearmans_results_scatter['value'])

spearmans_results_scatter['syn_methods'] = spearmans_results_scatter['variable'].str[:3]

strings_to_remove=('discount_rate','tax_rate', 'hours_run_per_day', 'electricity_cost', 'COD_foodwaste', 'N_foodwaste', 'carbon_COD_ratio', 'density_foodwaste', 'FW_landfill_decay_rate', 'landfill_years', 'Time_since_landfill_close', 'NaOH_amount', 'cost_NaOH_USA', 'COD_ppmill_sludge', 'N_ppmill_sludge', 'solids_mass_PPMS', 'specific_gravity_PPMS', 'PPMS_TS', 'ad_flowrate', 'organic_loading_rate', 'effectiveness_factor', 'methane_yield_COD', 'HRT', 'SRT', 'CHP_electricity_efficiency', 'Cp_sludge', 'U_roof', 'AD_bottom_slope', 'PPMS_COD_content', 'PPMS_VS_TS_ratio', 'AD_VS_destruction', 'AD_VS_destruction_mixed', 'PPMS_landfill_decay_rate',
                  'T_wall_10', 'T_floor_10', 'T_roof_10','T_wall_15', 'T_floor_15', 'T_roof_15','T_floor_20', 'T_wall_20', 'T_roof_20' , 'T_wall_25', 'T_floor_25', 'T_roof_25', 'AD_depth', 'AD_freeboard', 'COD_degredation', 'COD_degredation_mixed', 'dummy', 'FW_VS', 'FW_COD_content', 'CH4_GWP', 'NaOH_GWP', 'oil_GHG', 'hydro_GHG', 'bio_GHG', 'geo_GHG', 'wind_GHG', 'solar_GHG', 'nuclear_GHG', 'Electricity_cost_defunct', 'region', 'Tipping_fees_US_1','Tipping_fees_US_2','Tipping_fees_US_3','Tipping_fees_US_4','Tipping_fees_US_5','Tipping_fees_US_6', 'k_high_rain', 'k_low_rain', 'coal_use_US_1', 'oil_use_US_1', 'gas_use_US_1', 'hydro_use_US_1', 'bio_use_US_1', 'geo_use_US_1', 'wind_use_US_1', 'solar_use_US_1', 'nuclear_use_US_1', 'Electricity_cost_1', 'coal_use_US_2', 'oil_use_US_2', 'gas_use_US_2', 'hydro_use_US_2', 'bio_use_US_2', 'geo_use_US_2', 'wind_use_US_2', 'solar_use_US_2', 'nuclear_use_US_2', 'Electricity_cost_2', 'coal_use_US_3', 'oil_use_US_3', 'gas_use_US_3', 'hydro_use_US_3', 'bio_use_US_3', 'geo_use_US_3', 'wind_use_US_3', 'solar_use_US_3', 'nuclear_use_US_3', 'Electricity_cost_3', 'coal_use_US_4', 'oil_use_US_4', 'gas_use_US_4', 'hydro_use_US_4', 'bio_use_US_4', 'geo_use_US_4', 'wind_use_US_4', 'solar_use_US_4', 'nuclear_use_US_4', 'Electricity_cost_4', 'coal_use_US_5', 'oil_use_US_5', 'gas_use_US_5', 'hydro_use_US_5', 'bio_use_US_5', 'geo_use_US_5', 'wind_use_US_5', 'solar_use_US_5', 'nuclear_use_US_5', 'Electricity_cost_5', 'coal_use_US_6', 'oil_use_US_6', 'gas_use_US_6', 'hydro_use_US_6', 'bio_use_US_6', 'geo_use_US_6', 'wind_use_US_6', 'solar_use_US_6', 'nuclear_use_US_6', 'Electricity_cost_6', 'coal_use_US_7', 'oil_use_US_7', 'gas_use_US_7', 'hydro_use_US_7', 'bio_use_US_7', 'geo_use_US_7', 'wind_use_US_7', 'solar_use_US_7', 'nuclear_use_US_7','Transport_distance_US_1', 'Transport_distance_US_2', 'Transport_distance_US_3', 'Transport_distance_US_4', 'Transport_distance_US_5', 'Transport_distance_US_6', 'Specific_methane_yield_FW', 'Specific_methane_yield_mixed', 'CHP_heat_efficiency', 'Energy_content_methane', 'Density_PPMS', 'truck_capacity', 'diesel_transport', 'diesel_cost_US', 'gas_GHG', 'Specific_methane_yield_PPMS')

spearmans_results_scatter_filtered = spearmans_results_scatter[~spearmans_results_scatter['index'].isin(strings_to_remove)]

def convert_to_color(variable_name):
    color_mapping = dict(zip(spearmans_results_scatter['syn_methods'].unique(), colors))
    return color_mapping.get(variable_name, 'gray')  # Default to gray for unknown variables

# Add a new column 'Color' to the DataFrame
spearmans_results_scatter['color']= spearmans_results_scatter['syn_methods'].apply(convert_to_color)


#data filtering

strings_to_remove=('discount_rate','tax_rate', 'hours_run_per_day', 'electricity_cost', 'COD_foodwaste', 'N_foodwaste','carbon_COD_ratio', 'density_foodwaste', 'FW_landfill_decay_rate', 'landfill_years', 'Time_since_landfill_close', 'NaOH_amount', 'cost_NaOH_USA', 'COD_ppmill_sludge', 'N_ppmill_sludge', 'solids_mass_PPMS', 'specific_gravity_PPMS', 'PPMS_TS', 'ad_flowrate', 'organic_loading_rate', 'effectiveness_factor', 'methane_yield_COD', 'HRT', 'SRT', 'CHP_electricity_efficiency', 'Cp_sludge', 'U_roof', 'AD_bottom_slope', 'PPMS_COD_content', 'PPMS_VS_TS_ratio', 'AD_VS_destruction', 'AD_VS_destruction_mixed', 'PPMS_landfill_decay_rate',
                  'T_wall_10', 'T_floor_10', 'T_roof_10','T_wall_15', 'T_floor_15', 'T_roof_15','T_floor_20', 'T_wall_20', 'T_roof_20' , 'T_wall_25', 'T_floor_25', 'T_roof_25', 'AD_depth', 'AD_freeboard', 'COD_degredation', 'COD_degredation_mixed', 'dummy', 'FW_VS', 'FW_COD_content', 'CH4_GWP', 'NaOH_GWP', 'oil_GHG', 'hydro_GHG', 'bio_GHG', 'geo_GHG', 'wind_GHG', 'solar_GHG', 'nuclear_GHG', 'Electricity_cost_defunct', 'region', 'Tipping_fees_US_1','Tipping_fees_US_2','Tipping_fees_US_3','Tipping_fees_US_4','Tipping_fees_US_5','Tipping_fees_US_6', 'k_high_rain', 'k_low_rain', 'coal_use_US_1', 'oil_use_US_1', 'gas_use_US_1', 'hydro_use_US_1', 'bio_use_US_1', 'geo_use_US_1', 'wind_use_US_1', 'solar_use_US_1', 'nuclear_use_US_1', 'Electricity_cost_1', 'coal_use_US_2', 'oil_use_US_2', 'gas_use_US_2', 'hydro_use_US_2', 'bio_use_US_2', 'geo_use_US_2', 'wind_use_US_2', 'solar_use_US_2', 'nuclear_use_US_2', 'Electricity_cost_2', 'coal_use_US_3', 'oil_use_US_3', 'gas_use_US_3', 'hydro_use_US_3', 'bio_use_US_3', 'geo_use_US_3', 'wind_use_US_3', 'solar_use_US_3', 'nuclear_use_US_3', 'Electricity_cost_3', 'coal_use_US_4', 'oil_use_US_4', 'gas_use_US_4', 'hydro_use_US_4', 'bio_use_US_4', 'geo_use_US_4', 'wind_use_US_4', 'solar_use_US_4', 'nuclear_use_US_4', 'Electricity_cost_4', 'coal_use_US_5', 'oil_use_US_5', 'gas_use_US_5', 'hydro_use_US_5', 'bio_use_US_5', 'geo_use_US_5', 'wind_use_US_5', 'solar_use_US_5', 'nuclear_use_US_5', 'Electricity_cost_5', 'coal_use_US_6', 'oil_use_US_6', 'gas_use_US_6', 'hydro_use_US_6', 'bio_use_US_6', 'geo_use_US_6', 'wind_use_US_6', 'solar_use_US_6', 'nuclear_use_US_6', 'Electricity_cost_6', 'coal_use_US_7', 'oil_use_US_7', 'gas_use_US_7', 'hydro_use_US_7', 'bio_use_US_7', 'geo_use_US_7', 'wind_use_US_7', 'solar_use_US_7', 'nuclear_use_US_7','Transport_distance_US_1', 'Transport_distance_US_2', 'Transport_distance_US_3', 'Transport_distance_US_4', 'Transport_distance_US_5', 'Transport_distance_US_6', 'Specific_methane_yield_FW', 'Specific_methane_yield_mixed', 'CHP_heat_efficiency', 'Energy_content_methane', 'Density_PPMS', 'truck_capacity', 'diesel_transport', 'diesel_cost_US', 'gas_GHG', 'Specific_methane_yield_PPMS')


spearmans_results_scatter_filtered = spearmans_results_scatter[~spearmans_results_scatter['index'].isin(strings_to_remove)]


fig = plt.figure(figsize=(4,6))
ax = fig.add_subplot(111)
plt.xticks(rotation=90)

# Set the font name for axis tick labels to be Comic Sans
for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
for tick in ax.get_yticklabels():
    tick.set_fontname("Arial")

ax.grid(color = 'gray', linestyle = '--', linewidth = 0.5)

ax.set_xlabel("Sustainability indicators", fontname="Arial", fontsize=10)
ax.set_ylabel("Assumptions", fontname="Arial", fontsize=10) 
ax.scatter('variable','index', s='value', alpha=0.8, c='color', data=spearmans_results_scatter_filtered)
plt.tick_params(direction='inout')
plt.margins(y=.04)
#plt.savefig("spearmans_09022024.png", dpi=1000, bbox_inches="tight")
#plt.savefig("spearmans_09022024.eps", dpi=1000, bbox_inches="tight")
