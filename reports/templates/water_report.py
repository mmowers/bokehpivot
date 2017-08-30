'''
Static HTML report maker

To use, copy this file into the implementations/ directory and change the "path" and "base" variables below.
Run this file on command line with "python path/to/file.py"
'''
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import reeds_bokeh as rb

path = r'\\nrelqnap01d\ReEDS\FY17-WindRuns-MRM-d1c8e69-wind_cost_scenarios\runs\20170322_jedi_outputs'
base = 'ATB_Mid'
static_presets = [
    {'name': 'Total Withdrawals (Bil Gals)', 'result': 'Water Withdrawals (Bil Gals)', 'preset': 'Total'},
    {'name': 'Total Withdrawals Diff (Bil Gals)', 'result': 'Water Withdrawals (Bil Gals)', 'preset': 'Total', 'modify':'diff'},
    {'name': 'Base Huc Withdrawals (Bil Gals)', 'result': 'Water Withdrawals (Bil Gals)', 'preset': 'Explode scenario for huc', 'modify':'base_only'},
    {'name': 'Huc Withdrawals Diff (Bil Gals)', 'result': 'Water Withdrawals (Bil Gals)', 'preset': 'Explode scenario for huc', 'modify':'diff'},
    {'name': 'Total Consumption (Bil Gals)', 'result': 'Water Consumption (Bil Gals)', 'preset': 'Total'},
    {'name': 'Total Consumption Diff (Bil Gals)', 'result': 'Water Consumption (Bil Gals)', 'preset': 'Total', 'modify':'diff'},
    {'name': 'Base Huc Consumption (Bil Gals)', 'result': 'Water Consumption (Bil Gals)', 'preset': 'Explode scenario for huc', 'modify':'base_only'},
    {'name': 'Huc Consumption Diff (Bil Gals)', 'result': 'Water Consumption (Bil Gals)', 'preset': 'Explode scenario for huc', 'modify':'diff'},
]
rb.reeds_static(path, static_presets, base)
