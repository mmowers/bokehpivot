'''
Static HTML report maker

'''
from core import reeds_static

path = r'C:\Users\mmowers\Projects\Model Improvement\Bokeh\reeds_pivot_test'
base = 'Master'
static_presets = [
    {'result': 'Capacity (GW)', 'presets': ['Stacked Bars'], 'modify': 'base_only'},
    {'result': 'Capacity (GW)', 'presets': ['Stacked Bars'], 'modify': 'diff'},
    {'result': 'Capacity (GW)', 'presets': ['State Map 2050 Wind'], 'modify': 'base_only'},
    {'result': 'Capacity (GW)', 'presets': ['State Map 2050 Wind'], 'modify': 'diff'},
    {'result': 'Capacity (GW)', 'presets': ['State Map 2050 Solar'], 'modify': 'base_only'},
    {'result': 'Capacity (GW)', 'presets': ['State Map 2050 Solar'], 'modify': 'diff'},
    {'result': 'Generation (TWh)', 'presets': ['Stacked Bars'], 'modify': 'base_only'},
    {'result': 'Generation (TWh)', 'presets': ['Stacked Bars'], 'modify': 'diff'},
    {'result': 'Gen by m (GW)', 'presets': ['Stacked Bars 2050'], 'modify': 'base_only'},
    {'result': 'Gen by m (GW)', 'presets': ['Stacked Bars 2050'], 'modify': 'diff'},
    {'result': 'Emissions, Fuel, Prices', 'presets': ['CO2 Emissions (MMton)']},
    {'result': 'Emissions, Fuel, Prices', 'presets': ['CO2 Emissions (MMton)'], 'modify': 'diff'},
    {'result': 'Elec Price (2015$/MWh)', 'presets': ['National']},
    {'result': 'Elec Price (2015$/MWh)', 'presets': ['National'], 'modify': 'diff'},
    {'result': 'System Cost (2015$)', 'presets': ['2016-2050 Stacked Bars']},
    {'result': 'System Cost (2015$)', 'presets': ['2016-2050 Stacked Bars'], 'modify': 'diff'},
]
reeds_static(path, static_presets, base)
