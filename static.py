'''
Static HTML report maker

'''
from core import reeds_static

static_presets = [
    {'result': 'Capacity (GW)', 'presets': ['Stacked Capacity', 'State Map: 2050 Wind']},
    {'result': 'New Capacity (GW)', 'presets': ['Stacked Capacity']},
    {'result': 'Generation (TWh)', 'presets': ['Stacked Gen', 'State Map: 2050 Wind']},
]
path = r'C:\Users\mmowers\Projects\Model Improvement\Bokeh\reeds_pivot_test'
reeds_static(path, static_presets)
