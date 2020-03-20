import os
import pandas as pd

this_dir_path = os.path.dirname(os.path.realpath(__file__))
df_state_code = pd.read_csv(this_dir_path + '/../in/state_code.csv')
states = df_state_code['Code'].tolist()

static_presets = [
    {'name': 'Cumulative worldwide', 'config':{'x':'days from start', 'y':'number', 'series':'type', 'plot_width':'400', 'x_title':'Days since Jan 22', 'sync_axes':'No'}},
    {'name': 'Cumulative by country', 'config':{'chart_type':'Bar', 'x':'days from start', 'y':'number', 'series':'top countries', 'explode':'type', 'plot_width':'400', 'y_min':'0', 'x_title':'Days since Jan 22', 'sync_axes':'No'}},
    {'name': 'US cumulative by state past 2 weeks', 'config':{'chart_type':'Bar', 'x':'days ago', 'y':'number', 'series':'us top st', 'explode':'type', 'plot_width':'400', 'y_min':'0', 'x_title':'Days ago', 'sync_axes':'No', 'filter':{'country':['US'], 'days ago':list(range(15))}}},
    {'name': 'US cumulative deaths state map past 3 days', 'config':{'chart_type':'Area Map', 'x':'st', 'y':'number','explode':'days ago', 'filter':{'days ago':[0,1,2,3], 'country':['US'], 'st':states, 'type':['Deaths']}}},
]

# format_string = {'plot_width':'600', 'x_title':'Days since Jan 22'}
# for i in range(len(static_presets)):
#     static_presets[i]['config'].update(format_string)