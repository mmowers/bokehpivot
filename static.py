'''
Static HTML report maker

'''
from core import reeds_static

static_presets = [
    {'result': 'Capacity (GW)', 'presets': ['Stacked Capacity', 'State Map 2050 Wind', 'Stacked Difference']},
    {'result': 'New Capacity (GW)', 'presets': ['Stacked Capacity']},
    {'result': 'Generation (TWh)', 'presets': ['Stacked Gen', 'State Map 2050 Wind']},
]
path = r'C:\Users\mmowers\Projects\Model Improvement\Bokeh\reeds_pivot_test'
base = 'Master'
reeds_static(path, static_presets, base)
