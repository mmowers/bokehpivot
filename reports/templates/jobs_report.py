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
    {'name': 'Base Generation (TWh)', 'result': 'Generation (TWh)', 'preset': 'Stacked Bars'},
    {'name': 'Generation Diff (TWh)', 'result': 'Generation (TWh)', 'preset': 'Stacked Bars', 'modify': 'diff'},
    {'name': 'Base Capacity (GW)', 'result': 'Capacity (GW)', 'preset': 'Stacked Bars'},
    {'name': 'Capacity Diff (GW)', 'result': 'Capacity (GW)', 'preset': 'Stacked Bars', 'modify': 'diff'},
    {'name': 'Base New Capacity (GW, 2-year)', 'result': 'New Capacity (GW)', 'preset': 'Stacked Bars'},
    {'name': 'New Capacity Diff (GW, 2-year)', 'result': 'New Capacity (GW)', 'preset': 'Stacked Bars', 'modify': 'diff'},
    {'name': 'Jobs (millions)', 'result': 'JEDI Outputs', 'preset': 'Total Jobs'},
    {'name': 'Jobs Diff (millions)', 'result': 'JEDI Outputs', 'preset': 'Total Jobs', 'modify': 'diff'},
    {'name': 'Jobs (millions)', 'result': 'JEDI Outputs', 'preset': 'Stacked Jobs By Tech'},
    {'name': 'Jobs Diff (millions)', 'result': 'JEDI Outputs', 'preset': 'Stacked Jobs By Tech', 'modify': 'diff'},
    {'name': 'Earnings (Billion 2015$)', 'result': 'JEDI Outputs', 'preset': 'Stacked Earnings By Tech'},
    {'name': 'Earnings Diff (Billion 2015$)', 'result': 'JEDI Outputs', 'preset': 'Stacked Earnings By Tech', 'modify': 'diff'},
    {'name': 'Output (Billion 2015$)', 'result': 'JEDI Outputs', 'preset': 'Stacked Output By Tech'},
    {'name': 'Output Diff (Billion 2015$)', 'result': 'JEDI Outputs', 'preset': 'Stacked Output By Tech', 'modify': 'diff'},
    {'name': 'Value Add (Billion 2015$)', 'result': 'JEDI Outputs', 'preset': 'Stacked Value Add By Tech'},
    {'name': 'Value Add Diff (Billion 2015$)', 'result': 'JEDI Outputs', 'preset': 'Stacked Value Add By Tech', 'modify': 'diff'},
    {'name': 'Jobs (millions)', 'result': 'JEDI Outputs', 'preset': 'Stacked Jobs By Directness'},
    {'name': 'Jobs Diff (millions)', 'result': 'JEDI Outputs', 'preset': 'Stacked Jobs By Directness', 'modify': 'diff'},
    {'name': 'Jobs (millions)', 'result': 'JEDI Outputs', 'preset': 'Jobs By Directness'},
    {'name': 'Jobs Diff (millions)', 'result': 'JEDI Outputs', 'preset': 'Jobs By Directness', 'modify': 'diff'},
    {'name': 'Wind Jobs (millions)', 'result': 'JEDI Outputs', 'preset': 'Stacked Jobs By Directness', 'config': {'filter': {'jedi_tech':['wind'], 'metric':['jobs']}}},
    {'name': 'Wind Jobs Diff (millions)', 'result': 'JEDI Outputs', 'preset': 'Stacked Jobs By Directness', 'modify': 'diff', 'config': {'filter': {'jedi_tech':['wind'], 'metric':['jobs']}}},
    {'name': 'UPV Jobs (millions)', 'result': 'JEDI Outputs', 'preset': 'Stacked Jobs By Directness', 'config': {'filter': {'jedi_tech':['upv'], 'metric':['jobs']}}},
    {'name': 'UPV Jobs Diff (millions)', 'result': 'JEDI Outputs', 'preset': 'Stacked Jobs By Directness', 'modify': 'diff', 'config': {'filter': {'jedi_tech':['upv'], 'metric':['jobs']}}},
    {'name': 'Coal Jobs (millions)', 'result': 'JEDI Outputs', 'preset': 'Stacked Jobs By Directness', 'config': {'filter': {'jedi_tech':['coal'], 'metric':['jobs']}}},
    {'name': 'Coal Jobs Diff (millions)', 'result': 'JEDI Outputs', 'preset': 'Stacked Jobs By Directness', 'modify': 'diff', 'config': {'filter': {'jedi_tech':['coal'], 'metric':['jobs']}}},
    {'name': 'Gas Jobs (millions)', 'result': 'JEDI Outputs', 'preset': 'Stacked Jobs By Directness', 'config': {'filter': {'jedi_tech':['gas'], 'metric':['jobs']}}},
    {'name': 'Gas Jobs Diff (millions)', 'result': 'JEDI Outputs', 'preset': 'Stacked Jobs By Directness', 'modify': 'diff', 'config': {'filter': {'jedi_tech':['gas'], 'metric':['jobs']}}},
    {'name': 'Jobs (millions)', 'result': 'JEDI Outputs', 'preset': 'Stacked Jobs By Category'},
    {'name': 'Jobs Diff (millions)', 'result': 'JEDI Outputs', 'preset': 'Stacked Jobs By Category', 'modify': 'diff'},
    {'name': 'Average Onsite Jobs 2017-2050 (millions)', 'result': 'JEDI Outputs', 'preset': 'Average Onsite Jobs Map 2017-2050'},
    {'name': 'Average Onsite Jobs 2017-2050 Diff (millions)', 'result': 'JEDI Outputs', 'preset': 'Average Onsite Jobs Map 2017-2050', 'modify': 'diff'},
    {'name': 'Average Onsite Renewable Jobs 2017-2050 (millions)', 'result': 'JEDI Outputs', 'preset': 'Average Onsite Jobs Map 2017-2050', 'config':{'filter':{'jedi_tech':['wind','upv']}}},
    {'name': 'Average Onsite Renewable Jobs 2017-2050 Diff (millions)', 'result': 'JEDI Outputs', 'preset': 'Average Onsite Jobs Map 2017-2050', 'modify': 'diff', 'config':{'filter':{'jedi_tech':['wind','upv']}}},
    {'name': 'Average Onsite Fossil Jobs 2017-2050 (millions)', 'result': 'JEDI Outputs', 'preset': 'Average Onsite Jobs Map 2017-2050', 'config':{'filter':{'jedi_tech':['coal','gas']}}},
    {'name': 'Average Onsite Fossil Jobs 2017-2050 Diff (millions)', 'result': 'JEDI Outputs', 'preset': 'Average Onsite Jobs Map 2017-2050', 'modify': 'diff', 'config':{'filter':{'jedi_tech':['coal','gas']}}},
]
rb.reeds_static(path, static_presets, base)
