'''
This is an example of a custom static report with documenation on the different ways
to configure a report.
'''
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import reeds_bokeh as rb

path = r'\\nrelqnap01d\ReEDS\FY17-WindRuns-MRM-d1c8e69-wind_cost_scenarios\runs\20170322_jedi_outputs'

#base is not necessary, but it allows for 'modify' to be used below to show only the base, or difference with base
base = 'ATB_Mid'

static_presets = [
    #ReEDS presets may be used. 'result' and 'preset' values are required to match those in reeds.py. 'modify' (optional) may then
    #be used to take a difference with the base case ('diff') or to show base case only ('base_only')
    {'name': 'Base Capacity (GW)', 'result': 'Capacity (GW)', 'preset': 'Stacked Bars'},
    {'name': 'Capacity Diff (GW)', 'result': 'Capacity (GW)', 'preset': 'Stacked Bars', 'modify': 'diff'},

    #Config may be entered directly as well, rather than via ReEDS presets. For instance, the above "Capacity Diff" may be entered as:
    {'name':
        'Capacity Diff 2 (GW)',
     'config':
        {'result':'Capacity (GW)', 'chart_type':'Bar', 'x':'year', 'y':'Capacity (GW)', 'series':'tech', 'explode':'scenario',
        'adv_op':'Difference', 'adv_col':'scenario', 'adv_col_base':base, 'bar_width':r'1.75', 'filter': {}}
    },
    #I made this by using the Export URL/Config button on the bokehpivot interface and pasting the object as the value of 'config'.
    #Note that the result is way longer.

    #Additional config may also be added on top of ReEDS presets like so:
    {'name': 'Capacity Diff (GW)', 'result': 'Capacity (GW)', 'preset': 'Stacked Bars', 'modify': 'diff',
     'config': {'bar_width':'8', 'filter': {'year':['2020','2030','2040','2050'], }}},
]


rb.reeds_static(path, static_presets, base=base, report_format='both')
'''
Notes on the function call:
    -core.static_report() may be called instead of reeds_bokeh.reeds_static(), but this disallows any elements above
        that use the top-level 'result', 'preset', or 'modify' keys, which are reeds-specific, and 'base' is also not an argument to core.static_report().
    -'report_format' allows 'excel', 'html', or 'both'
    - base is necessary if 'modify' is used above
'''
