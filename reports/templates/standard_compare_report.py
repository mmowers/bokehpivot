'''
Static HTML report maker

To use, copy this file into the implementations/ directory, change path and base, and run
'''
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import reeds_bokeh as rb

path = r'\\nrelqnap01d\ReEDS\FY17-WindRuns-MRM-d1c8e69-wind_cost_scenarios\runs\20170322_jedi_outputs'
base = 'ATB_Mid'
static_presets = [
    {'name': 'Base Capacity (GW)', 'result': 'Capacity (GW)', 'preset': 'Stacked Bars', 'modify': 'base_only'},
    {'name': 'Capacity Diff (GW)', 'result': 'Capacity (GW)', 'preset': 'Stacked Bars', 'modify': 'diff'},
    {'name': '2050 Base Wind Capacity (GW)', 'result': 'Capacity (GW)', 'preset': 'State Map 2050 Wind', 'modify': 'base_only'},
    {'name': '2050 Wind Capacity Diff (GW)', 'result': 'Capacity (GW)', 'preset': 'State Map 2050 Wind', 'modify': 'diff'},
    {'name': '2050 Base Solar Capacity (GW)', 'result': 'Capacity (GW)', 'preset': 'State Map 2050 Solar', 'modify': 'base_only'},
    {'name': '2050 Solar Capacity Diff (GW)', 'result': 'Capacity (GW)', 'preset': 'State Map 2050 Solar', 'modify': 'diff'},
    {'name': 'Base Generation (TWh)', 'result': 'Generation (TWh)', 'preset': 'Stacked Bars', 'modify': 'base_only'},
    {'name': 'Generation Diff (TWh)', 'result': 'Generation (TWh)', 'preset': 'Stacked Bars', 'modify': 'diff'},
    {'name': '2050 Base Gen by m (GW)', 'result': 'Gen by m (GW)', 'preset': 'Stacked Bars 2050', 'modify': 'base_only'},
    {'name': '2050 Gen by m Diff (GW)', 'result': 'Gen by m (GW)', 'preset': 'Stacked Bars 2050', 'modify': 'diff'},
    {'name': 'Base CO2 Emissions (MMton)', 'result': 'Emissions, Fuel, Prices', 'preset': 'CO2 Emissions (MMton)'},
    {'name': 'CO2 Emissions Diff (MMton)', 'result': 'Emissions, Fuel, Prices', 'preset': 'CO2 Emissions (MMton)', 'modify': 'diff'},
    {'name': 'Elec Price (2015$/MWh)', 'result': 'Elec Price (2015$/MWh)', 'preset': 'National'},
    {'name': 'Elec Price Diff (2015$/MWh)', 'result': 'Elec Price (2015$/MWh)', 'preset': 'National', 'modify': 'diff'},
    {'name': 'System Cost (Bil 2015$)', 'result': 'Sys Cost (Bil 2015$)', 'preset': '2017-2050 Stacked Bars'},
    {'name': 'System Cost Diff (Bil 2015$)', 'result': 'Sys Cost (Bil 2015$)', 'preset': '2017-2050 Stacked Bars', 'modify': 'diff'},
]
rb.reeds_static(path, static_presets, base)