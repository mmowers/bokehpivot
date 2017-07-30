'''
Static HTML report maker

To use, copy this file into the implementations/ directory, change path, and run
'''
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import reeds_bokeh as rb

path = r'\\nrelqnap01d\ReEDS\FY17-WindRuns-MRM-d1c8e69-wind_cost_scenarios\runs\20170322_jedi_outputs\ATB_Mid'
static_presets = [
    {'name': 'Capacity (GW)', 'result': 'Capacity (GW)', 'preset': 'Stacked Bars'},
    {'name': 'New Capacity (GW)', 'result': 'New Capacity (GW)', 'preset': 'Stacked Bars'},
    {'name': 'Retirements (GW)', 'result': 'Retirements (GW)', 'preset': 'Stacked Bars'},
    {'name': '2050 Wind Capacity (GW)', 'result': 'Wind Capacity (GW)', 'preset': '2050 Map'},
    {'name': '2050 Solar Capacity (GW)', 'result': 'Capacity (GW)', 'preset': 'PCA Map 2050 Solar'},
    {'name': 'Generation (TWh)', 'result': 'Generation (TWh)', 'preset': 'Stacked Bars'},
    {'name': '2050 Gen by timeslice (GW)', 'result': 'Gen by m (GW)', 'preset': 'Stacked Bars 2050'},
    {'name': 'CO2 Emissions (MMton)', 'result': 'Emissions, Fuel, Prices', 'preset': 'CO2 Emissions (MMton)'},
    {'name': 'Elec Price (2015$/MWh)', 'result': 'Elec Price (2015$/MWh)', 'preset': 'National Scenario'},
    {'name': '2017-2050 System Cost (Bil 2015$)', 'result': 'Sys Cost (Bil 2015$)', 'preset': '2017-2050 Stacked Bars'},
]
rb.reeds_static(path, static_presets)
