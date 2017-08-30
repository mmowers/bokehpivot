'''
Static HTML report maker

To use, copy this file into the implementations/ directory and change the "path" and "base" variables below.
Run this file on command line with "python path/to/file.py"
'''
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import reeds_bokeh as rb

path = r'\\nrelqnap01d\ReEDS\FY17-JEDI-MRM-jedi\runs\2017-08-22 runs'
base = 'Base'
static_presets = [
    {'name': 'Base Generation (TWh)', 'result': 'Generation (TWh)', 'preset': 'Stacked Bars', 'modify': 'base_only'},
    {'name': 'Generation Diff (TWh)', 'result': 'Generation (TWh)', 'preset': 'Stacked Bars', 'modify': 'diff'},
    {'name': 'Base Capacity (GW)', 'result': 'Capacity (GW)', 'preset': 'Stacked Bars', 'modify': 'base_only'},
    {'name': 'Capacity Diff (GW)', 'result': 'Capacity (GW)', 'preset': 'Stacked Bars', 'modify': 'diff'},
    {'name': 'Base New Capacity (GW)', 'result': 'New Capacity (GW)', 'preset': 'Stacked Bars', 'modify': 'base_only'},
    {'name': 'New Capacity Diff (GW)', 'result': 'New Capacity (GW)', 'preset': 'Stacked Bars', 'modify': 'diff'},
    {'name': 'System Cost (Bil 2015$)', 'result': 'Sys Cost (Bil 2015$)', 'preset': '2017-2050 Stacked Bars'},
    {'name': 'System Cost Diff (Bil 2015$)', 'result': 'Sys Cost (Bil 2015$)', 'preset': '2017-2050 Stacked Bars', 'modify': 'diff'},
    {'name': 'JEDI metrics', 'result': 'JEDI Outputs', 'preset': 'Main Metrics'},
    {'name': 'JEDI metrics diff', 'result': 'JEDI Outputs', 'preset': 'Main Metrics', 'modify': 'diff'},
    {'name': 'Jobs By Tech', 'result': 'JEDI Outputs', 'preset': 'Jobs By Tech'},
    {'name': 'Jobs By Tech diff', 'result': 'JEDI Outputs', 'preset': 'Jobs By Tech', 'modify': 'diff'},
    {'name': '2050 Direct Jobs Base', 'result': 'JEDI Outputs', 'preset': '2050 Direct Jobs Map'},
    {'name': '2050 Direct Jobs Diff', 'result': 'JEDI Outputs', 'preset': '2050 Direct Jobs Map', 'modify': 'diff'},
]
rb.reeds_static(path, static_presets, base)
