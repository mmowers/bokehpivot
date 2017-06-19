'''
Static HTML report maker

'''
from core import reeds_static

static_presets = [
    {'result': 'Capacity (GW)', 'presets': ['Stacked Capacity', 'State Map 2050 Wind', 'Stacked Difference']},
    {'result': 'New Capacity (GW)', 'presets': ['Stacked Capacity']},
    {'result': 'Generation (TWh)', 'presets': ['Stacked Gen', 'State Map 2050 Wind']},
]
path = r'\\nrelqnap01d\ReEDS\FY17-ABB Update-MRM-a877bcf-abb_update\runs'
base = 'Master'
reeds_static(path, static_presets, base)
