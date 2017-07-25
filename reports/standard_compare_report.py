'''
Static HTML report maker

'''
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import reeds_bokeh as rb

path = r'\\nrelqnap01d\ReEDS\FY17-WindRuns-MRM-d1c8e69-wind_cost_scenarios\runs\20170322_jedi_outputs'
base = 'ATB_Mid'
static_presets = [
    {'result': 'Capacity (GW)', 'preset': 'Stacked Bars', 'modify': 'base_only'},
    {'result': 'Capacity (GW)', 'preset': 'Stacked Bars', 'modify': 'diff'},
    {'result': 'Capacity (GW)', 'preset': 'State Map 2050 Wind', 'modify': 'base_only'},
    {'result': 'Capacity (GW)', 'preset': 'State Map 2050 Wind', 'modify': 'diff'},
    {'result': 'Capacity (GW)', 'preset': 'State Map 2050 Solar', 'modify': 'base_only'},
    {'result': 'Capacity (GW)', 'preset': 'State Map 2050 Solar', 'modify': 'diff'},
    {'result': 'Generation (TWh)', 'preset': 'Stacked Bars', 'modify': 'base_only'},
    {'result': 'Generation (TWh)', 'preset': 'Stacked Bars', 'modify': 'diff'},
    {'result': 'Gen by m (GW)', 'preset': 'Stacked Bars 2050', 'modify': 'base_only'},
    {'result': 'Gen by m (GW)', 'preset': 'Stacked Bars 2050', 'modify': 'diff'},
    {'result': 'Emissions, Fuel, Prices', 'preset': 'CO2 Emissions (MMton)'},
    {'result': 'Emissions, Fuel, Prices', 'preset': 'CO2 Emissions (MMton)', 'modify': 'diff'},
    {'result': 'Elec Price (2015$/MWh)', 'preset': 'National'},
    {'result': 'Elec Price (2015$/MWh)', 'preset': 'National', 'modify': 'diff'},
    {'result': 'Sys Cost (Bil 2015$)', 'preset': '2017-2050 Stacked Bars'},
    {'result': 'Sys Cost (Bil 2015$)', 'preset': '2017-2050 Stacked Bars', 'modify': 'diff'},
]
rb.reeds_static(path, static_presets, base)
