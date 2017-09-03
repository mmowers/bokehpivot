'''
These are examples of custom configurations that can be added directly to create_report.py
'''

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
