'''
Static HTML report maker

'''
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import reeds_bokeh as rb

path = r'\\nrelqnap01d\ReEDS\FY17-WindRuns-MRM-d1c8e69-wind_cost_scenarios\runs\20170322_jedi_outputs\ATB_Mid'
static_presets = [
    {'result': 'Capacity (GW)', 'presets': ['Stacked Bars']},
    {'result': 'New Capacity (GW)', 'presets': ['Stacked Bars']},
]
rb.reeds_static(path, static_presets)
