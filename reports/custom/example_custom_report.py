'''
Static HTML report maker

'''
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import core

path = r'C:\Users\mmowers\Projects\Model Improvement\Bokeh\reeds_pivot_test\Master'
static_presets = [
    {'result': 'Capacity (GW)', 'presets': ['Stacked Bars']},
    {'result': 'New Capacity (GW)', 'presets': ['Stacked Bars']},
    {'result': 'Retirements (GW)', 'presets': ['Stacked Bars']},
]
core.reeds_static(path, static_presets)
