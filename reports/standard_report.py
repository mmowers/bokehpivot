'''
Static HTML report maker

'''
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import reeds_bokeh as rb

path = r'\\nrelqnap01d\ReEDS\FY17-WindRuns-MRM-d1c8e69-wind_cost_scenarios\runs\20170322_jedi_outputs\ATB_Mid'
static_presets = [
    {'result': 'Capacity (GW)', 'preset': 'Stacked Bars'},
    {'result': 'New Capacity (GW)', 'preset': 'Stacked Bars'},
    {'result': 'Retirements (GW)', 'preset': 'Stacked Bars'},
    {'result': 'Wind Capacity (GW)', 'preset': '2050 Map'},
    {'result': 'Capacity (GW)', 'preset': 'PCA Map 2050 Solar'},
    {'result': 'Generation (TWh)', 'preset': 'Stacked Bars'},
    {'result': 'Gen by m (GW)', 'preset': 'Stacked Bars 2050'},
    {'result': 'Emissions, Fuel, Prices', 'preset': 'CO2 Emissions (MMton)'},
    {'result': 'Elec Price (2015$/MWh)', 'preset': 'National Scenario'},
    {'result': 'Sys Cost (Bil 2015$)', 'preset': '2017-2050 Stacked Bars'},
]
rb.reeds_static(path, static_presets)
