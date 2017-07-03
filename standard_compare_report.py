'''
Static HTML report maker

'''
from core import reeds_static

path = r'C:\Users\mmowers\Projects\Model Improvement\Bokeh\reeds_pivot_test'
base = 'Master'
static_presets = [
    {'result': 'Capacity (GW)', 'presets': ['Stacked Bars'], 'base_only': True},
    {'result': 'Capacity (GW)', 'presets': ['Stacked Difference']},
]
reeds_static(path, static_presets, base)
