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
    # {'result': 'Capacity (GW)', 'presets': ['Difference Bars']},

    # {'result': 'New Capacity (GW)', 'presets': ['Stacked Bars']},
    # {'result': 'Retirements (GW)', 'presets': ['Stacked Bars']},
    # {'result': 'Capacity (GW)', 'presets': ['PCA Map 2050 Solar']},
    # {'result': 'Generation (TWh)', 'presets': ['Stacked Bars']},
    # {'result': 'Gen by m (GW)', 'presets': ['Stacked Bars 2050']},
    # {'result': 'Emissions, Fuel, Prices', 'presets': ['CO2 Emissions (MMton)']},
    # {'result': 'Elec Price (2015$/MWh)', 'presets': ['National Scenario']},
    # {'result': 'System Cost (2015$)', 'presets': ['2016-2050 Stacked Bars']},


    #Plan (4 scenarios):
    #Capacity bars (4)
    #Capacity diff charts (3)
    #Capacity maps (wind, pv) (state)
    #Capacity diff maps, wind (state)
    #Gen Bars
    #Gen diff bars
    #System costs
    #System cost diffs
    #Electricity price
    #Elec price diff
    #CO2 emissions
    #CO2 emissions diff
]
reeds_static(path, static_presets, base)
