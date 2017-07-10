'''
Static HTML report maker

'''
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import core

path = r'\\nrelqnap01d\ReEDS\FY17-WindRuns-MRM-d1c8e69-wind_cost_scenarios\runs\20170322_jedi_outputs\ATB_Mid'
static_presets = [
    {'result': 'Capacity (GW)', 'presets': ['Stacked Bars']},
    {'result': 'New Capacity (GW)', 'presets': ['Stacked Bars']},
    {'result': 'Retirements (GW)', 'presets': ['Stacked Bars']},
    {'result': 'Wind Capacity (GW)', 'presets': ['2050 Map']},
    {'result': 'Capacity (GW)', 'presets': ['PCA Map 2050 Solar']},
    {'result': 'Generation (TWh)', 'presets': ['Stacked Bars']},
    {'result': 'Gen by m (GW)', 'presets': ['Stacked Bars 2050']},
    {'result': 'Emissions, Fuel, Prices', 'presets': ['CO2 Emissions (MMton)']},
    {'result': 'Elec Price (2015$/MWh)', 'presets': ['National Scenario']},
    {'result': 'Sys Cost (Bil 2015$)', 'presets': ['2017-2050 Stacked Bars']},
]
core.reeds_static(path, static_presets)
