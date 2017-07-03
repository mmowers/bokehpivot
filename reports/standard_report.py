'''
Static HTML report maker

'''
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import core

path = r'C:\Users\mmowers\Projects\Model Improvement\Bokeh\reeds_pivot_test\Master'
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
    {'result': 'System Cost (2015$)', 'presets': ['2016-2050 Stacked Bars']},
]
core.reeds_static(path, static_presets)
