'''
Pivot chart maker for CSVs, GDX files, and ReEDS run results.

'''
from __future__ import division
import os
import math
import json
import pandas as pd
import collections
import bokeh.io as bio
import bokeh.layouts as bl
import bokeh.models.widgets as bmw
import bokeh.models.sources as bms
import bokeh.models.tools as bmt
import bokeh.plotting as bp
import datetime
import six.moves.urllib.parse as urlp
import gdx2py
from reeds import results_meta, columns_meta

#Defaults to configure:
PLOT_WIDTH = 300
PLOT_HEIGHT = 300
PLOT_FONT_SIZE = 10
PLOT_AXIS_LABEL_SIZE = 8
PLOT_LABEL_ORIENTATION = 45
OPACITY = 0.8
X_SCALE = 1
Y_SCALE = 1
CIRCLE_SIZE = 9
BAR_WIDTH = 0.5
LINE_WIDTH = 2
COLORS = ['#5e4fa2', '#3288bd', '#66c2a5', '#abdda4', '#e6f598', '#fee08b', '#fdae61', '#f46d43', '#d53e4f', '#9e0142']*1000
C_NORM = "#31AADE"
CHARTTYPES = ['Dot', 'Line', 'Bar', 'Area', 'Map']
STACKEDTYPES = ['Bar', 'Area']
AGGREGATIONS = ['None', 'Sum', 'Ave', 'Weighted Ave']
ADV_BASES = ['Consecutive', 'Total']
MAP_NUM_BINS = 5

#List of widgets that use columns as their selectors
WDG_COL = ['x', 'y', 'x_group', 'series', 'explode', 'explode_group']

#List of widgets that don't use columns as selector and share general widget update function
WDG_NON_COL = ['chart_type', 'y_agg', 'y_weight', 'adv_op', 'adv_col_base', 'plot_title', 'plot_title_size',
    'plot_width', 'plot_height', 'opacity', 'x_min', 'x_max', 'x_scale', 'x_title',
    'x_title_size', 'x_major_label_size', 'x_major_label_orientation',
    'y_min', 'y_max', 'y_scale', 'y_title', 'y_title_size', 'y_major_label_size',
    'circle_size', 'bar_width', 'line_width', 'map_num', 'map_min', 'map_max']

#initialize globals dict for variables that are modified within update functions.
GL = {'df_source':None, 'df_plots':None, 'columns':None, 'data_source_wdg':None, 'variant_wdg':None, 'widgets':None, 'controls': None, 'plots':None}

#ReEDS globals
custom_sorts = {} #keys are column names. Values are lists of values in the desired sort order
scenarios = [] #each element is a dict with name of scenario and path to scenario
result_dfs = {} #keys are ReEDS result names. Values are dataframes for that result (with 'scenario' as one of the columns)

#os globals
this_dir_path = os.path.dirname(os.path.realpath(__file__))

def initialize():
    '''
    On initial load, read 'widgets' parameter from URL query string and use to set data source (data_source)
    and widget configuration object (wdg_config). Initialize controls and plots areas of layout, and
    send data to opened browser.
    '''
    print('***Initializing...')
    wdg_config = {}
    args = bio.curdoc().session_context.request.arguments
    wdg_arr = args.get('widgets')
    data_source = ''
    if wdg_arr is not None:
        wdg_config = json.loads(urlp.unquote(wdg_arr[0].decode('utf-8')))
        if 'data' in wdg_config:
            data_source = str(wdg_config['data'])

    #build widgets and plots
    GL['data_source_wdg'] = build_data_source_wdg(data_source)
    GL['controls'] = bl.widgetbox(list(GL['data_source_wdg'].values()), id='widgets_section')
    GL['plots'] = bl.column([], id='plots_section')
    layout = bl.row(GL['controls'], GL['plots'], id='layout')

    if data_source != '':
        update_data_source(init_load=True, init_config=wdg_config)
        set_wdg_col_options()
        update_plots()

    bio.curdoc().add_root(layout)
    bio.curdoc().title = "Exploding Pivot Chart Maker"
    print('***Done Initializing')

def build_data_source_wdg(data_source):
    '''
    Return the initial data source widget, prefilled with an input data_source
    Args:
        data_source (string): Path to data source
    Returns:
        wdg (ordered dict): ordered dictionary of bokeh.models.widgets (in this case only one) for data source.
    '''
    wdg = collections.OrderedDict()
    wdg['data_dropdown'] = bmw.Div(text='Data Source (required)', css_classes=['data-dropdown'])
    wdg['data'] = bmw.TextInput(value=data_source, css_classes=['wdgkey-data', 'data-drop'])
    wdg['data'].on_change('value', update_data)
    return wdg

def get_df_csv(data_source):
    '''
    Read a csv into a pandas dataframe, and determine which columns of the dataframe
    are discrete (strings), continuous (numbers), able to be filtered (aka filterable),
    and able to be used as a series (aka seriesable). NA values are filled based on the type of column,
    and the dataframe and columns are returned.

    Args:
        data_source (string): Path to csv file.

    Returns:
        df_source (pandas dataframe): A dataframe of the csv source, with filled NA values.
        cols (dict): Keys are categories of columns of df_source, and values are a list of columns of that category.
    '''
    print('***Fetching csv...')
    df_source = pd.read_csv(data_source)
    cols = {}
    cols['all'] = df_source.columns.values.tolist()
    cols['discrete'] = [x for x in cols['all'] if df_source[x].dtype == object]
    cols['continuous'] = [x for x in cols['all'] if x not in cols['discrete']]
    cols['x-axis'] = cols['all']
    cols['y-axis'] = cols['continuous']
    cols['filterable'] = cols['discrete']+[x for x in cols['continuous'] if len(df_source[x].unique()) < 100]
    cols['seriesable'] = cols['discrete']+[x for x in cols['continuous'] if len(df_source[x].unique()) < 60]
    df_source[cols['discrete']] = df_source[cols['discrete']].fillna('{BLANK}')
    df_source[cols['continuous']] = df_source[cols['continuous']].fillna(0)
    print('***Done fetching csv.')
    return (df_source, cols)

def get_wdg_gdx(data_source):
    '''
    Create a parameter select widget and return it.

    Args:
        data_source (string): Path to gdx file.

    Returns:
        topwdg (ordered dict): Dictionary of bokeh.model.widgets.
    '''
    return #need to implement!

def get_wdg_reeds(path, init_load=False, wdg_config={}):
    '''
    From data source path, fetch paths to scenarios and return dict of widgets for
    meta files, scenarios, and results

    Args:
        path (string): Path to a ReEDS run folder or a folder containing ReEDS runs folders.
        init_load (Boolean): True if this is the initial page load. False otherwise.
        wdg_config (dict): initial configuration for widgets.

    Returns:
        topwdg (ordered dict): Dictionary of bokeh.model.widgets.
    '''
    print('***Fetching ReEDS scenarios...')
    topwdg = collections.OrderedDict()

    #Meta widgets
    topwdg['meta'] = bmw.Div(text='Meta', css_classes=['meta-dropdown'])
    for col in columns_meta:
        if 'map' in columns_meta[col]:
            topwdg['meta_map_'+col] = bmw.TextInput(title='"'+col+ '" Map', value=columns_meta[col]['map'], css_classes=['wdgkey-meta_map_'+col, 'meta-drop'])
            if init_load and 'meta_map_'+col in wdg_config: topwdg['meta_map_'+col].value = str(wdg_config['meta_map_'+col])
            topwdg['meta_map_'+col].on_change('value', update_reeds_meta)
        if 'join' in columns_meta[col]:
            topwdg['meta_join_'+col] = bmw.TextInput(title='"'+col+ '" Join', value=columns_meta[col]['join'], css_classes=['wdgkey-meta_join_'+col, 'meta-drop'])
            if init_load and 'meta_join_'+col in wdg_config: topwdg['meta_join_'+col].value = str(wdg_config['meta_join_'+col])
            topwdg['meta_join_'+col].on_change('value', update_reeds_meta)
        if 'style' in columns_meta[col]:
            topwdg['meta_style_'+col] = bmw.TextInput(title='"'+col+ '" Style', value=columns_meta[col]['style'], css_classes=['wdgkey-meta_style_'+col, 'meta-drop'])
            if init_load and 'meta_style_'+col in wdg_config: topwdg['meta_style_'+col].value = str(wdg_config['meta_style_'+col])
            topwdg['meta_style_'+col].on_change('value', update_reeds_meta)

    #Filter Scenarios widgets and Result widget
    scenarios[:] = []
    runs_paths = path.split('|')
    for runs_path in runs_paths:
        runs_path = runs_path.strip()
        #if the path is pointing to a csv file, gather all scenarios from that file
        if os.path.isfile(runs_path) and runs_path.lower().endswith('.csv'):
            custom_sorts['scenario'] = []
            abs_path = str(os.path.abspath(runs_path))
            df_scen = pd.read_csv(abs_path)
            for i_scen, scen in df_scen.iterrows():
                if os.path.isdir(scen['path']):
                    abs_path_scen = os.path.abspath(scen['path'])
                    if os.path.isdir(abs_path_scen+'/gdxfiles'):
                        custom_sorts['scenario'].append(scen['name'])
                        scenarios.append({'name': scen['name'], 'path': abs_path_scen})
        #Else if the path is pointing to a directory, check if the directory is a run folder
        #containing gdxfiles/ and use this as the lone scenario. Otherwise, it must contain
        #run folders, so gather all of those scenarios.
        elif os.path.isdir(runs_path):
            abs_path = str(os.path.abspath(runs_path))
            if os.path.isdir(abs_path+'/gdxfiles'):
                scenarios.append({'name': os.path.basename(abs_path), 'path': abs_path})
            else:
                subdirs = os.walk(abs_path).next()[1]
                for subdir in subdirs:
                    if os.path.isdir(abs_path+'/'+subdir+'/gdxfiles'):
                        abs_subdir = str(os.path.abspath(abs_path+'/'+subdir))
                        scenarios.append({'name': subdir, 'path': abs_subdir})
    #If we have scenarios, build widgets for scenario filters and result.
    for key in ["filter_scenarios_dropdown", "filter_scenarios", "result"]:
        topwdg.pop(key, None)
    if scenarios:
        labels = [a['name'] for a in scenarios]
        topwdg['filter_scenarios_dropdown'] = bmw.Div(text='Filter Scenarios', css_classes=['filter-scenarios-dropdown'])
        topwdg['filter_scenarios'] = bmw.CheckboxGroup(labels=labels, active=list(range(len(labels))), css_classes=['wdgkey-filter_scenarios'])
        if init_load and 'filter_scenarios' in wdg_config: topwdg['filter_scenarios'].active = wdg_config['filter_scenarios']
        topwdg['result'] = bmw.Select(title='Result', value='None', options=['None']+list(results_meta.keys()), css_classes=['wdgkey-result'])
        if init_load and 'result' in wdg_config: topwdg['result'].value = str(wdg_config['result'])
        topwdg['result'].on_change('value', update_reeds_result)
    print('***Done fetching ReEDS scenarios.')
    return topwdg

def get_reeds_data(topwdg):
    '''
    For a selected ReEDS result and set of scenarios, fetch gdx data,
    preprocess it, and add to global result_dfs dictionary if the data
    hasn't already been fetched.

    Args:
        topwdg (ordered dict): ReEDS widgets (meta widgets, scenarios widget, result widget)

    Returns:
        Nothing. Global result_dfs is modified
    '''
    result = topwdg['result'].value
    print('***Fetching ' + str(result) + ' for selected scenarios...')
    #A result has been selected, so either we retrieve it from result_dfs,
    #which is a dict with one dataframe for each result, or we make a new key in the result_dfs
    if result not in result_dfs:
            result_dfs[result] = None
            cur_scenarios = []
    else:
        cur_scenarios = result_dfs[result]['scenario'].unique().tolist() #the scenarios that have already been retrieved and stored in result_dfs
    #For each selected scenario, retrieve the data from gdx if we don't already have it,
    #and update result_dfs with the new data.
    result_meta = results_meta[result]
    for i in topwdg['filter_scenarios'].active:
        scenario_name = scenarios[i]['name']
        if scenario_name not in cur_scenarios:
            #get the gdx result and preprocess
            if 'sources' in result_meta:
                #If we have multiple parameters as data sources, we must gather them all, and the first preprocess
                #function (which is necessary) will accept a dict of dataframes and return a combined dataframe.
                df_scen_result = {}
                for src in result_meta['sources']:
                    data = gdx2py.par2list(scenarios[i]['path'] + '\\gdxfiles\\' + src['file'], src['param'])
                    df_src = pd.DataFrame(data)
                    df_src.columns = src['columns']
                    df_scen_result[src['name']] = df_src
            else:
                #else we have only one parameter as a data source
                data = gdx2py.par2list(scenarios[i]['path'] + '\\gdxfiles\\' + result_meta['file'], result_meta['param'])
                df_scen_result = pd.DataFrame(data)
                df_scen_result.columns = result_meta['columns']
            if 'preprocess' in result_meta:
                for preprocess in result_meta['preprocess']:
                    df_scen_result = preprocess['func'](df_scen_result, **preprocess['args'])
            df_scen_result['scenario'] = scenario_name
            if result_dfs[result] is None:
                result_dfs[result] = df_scen_result
            else:
                result_dfs[result] = pd.concat([result_dfs[result], df_scen_result]).reset_index(drop=True)
        print('***Done fetching ' + str(result) + ' for ' + str(scenario_name) + '.')
    print('***Done fetching ' + str(result) + '.')

def process_reeds_data(topwdg):
    '''
    Apply joins, mappings, ordering data to a selected result dataframe.
    Also categorize the columns of the dataframe and fill NA values.

    Args:
        topwdg (ordered dict): ReEDS widgets (meta widgets, scenarios widget, result widget)

    Returns:
        df (pandas dataframe): A dataframe of the ReEDS result, with filled NA values.
        cols (dict): Keys are categories of columns of df_source, and values are a list of columns of that category.
    '''
    print('***Apply joins, maps, ordering to ReEDS data...')
    df = result_dfs[topwdg['result'].value].copy()
    #apply joins
    for col in df.columns.values.tolist():
        if 'meta_join_'+col in topwdg and topwdg['meta_join_'+col].value != '':
            df_join = pd.read_csv(topwdg['meta_join_'+col].value.replace('"',''))
            #remove columns to left of col in df_join
            for c in df_join.columns.values.tolist():
                if c == col:
                    break
                df_join.drop(c, axis=1, inplace=True)
            #remove duplicate rows
            df_join.drop_duplicates(subset=col, inplace=True)
            #merge df_join into df
            df = pd.merge(left=df, right=df_join, on=col, sort=False)

    #apply mappings
    for col in df.columns.values.tolist():
        if 'meta_map_'+col in topwdg and topwdg['meta_map_'+col].value != '':
            df_map = pd.read_csv(topwdg['meta_map_'+col].value.replace('"',''))
            #filter out values that aren't in raw column
            df = df[df[col].isin(df_map['raw'].values.tolist())]
            #now map from raw to display
            map_dict = dict(zip(list(df_map['raw']), list(df_map['display'])))
            df[col] = df[col].map(map_dict)

    #apply custom styling
    for col in df.columns.values.tolist():
        if 'meta_style_'+col in topwdg and topwdg['meta_style_'+col].value != '':
            df_style = pd.read_csv(topwdg['meta_style_'+col].value.replace('"',''))
            #filter out values that aren't in order column
            df = df[df[col].isin(df_style['order'].values.tolist())]
            #add to custom_sorts with new order
            custom_sorts[col] = df_style['order'].tolist()
    cols = {}
    cols['all'] = df.columns.values.tolist()
    for c in cols['all']:
        if c in columns_meta:
            if columns_meta[c]['type'] is 'number':
                df[c] = pd.to_numeric(df[c], errors='coerce')
            elif columns_meta[c]['type'] is 'string':
                df[c] = df[c].astype(str)

    cols['discrete'] = [x for x in cols['all'] if df[x].dtype == object]
    cols['continuous'] = [x for x in cols['all'] if x not in cols['discrete']]
    cols['y-axis'] = [x for x in cols['continuous'] if x not in columns_meta or columns_meta[x]['y-allow']]
    cols['x-axis'] = [x for x in cols['all'] if x not in cols['y-axis']]
    cols['filterable'] = cols['discrete']+[x for x in cols['continuous'] if x in columns_meta and columns_meta[x]['filterable']]
    cols['seriesable'] = cols['discrete']+[x for x in cols['continuous'] if x in columns_meta and columns_meta[x]['seriesable']]
    df[cols['discrete']] = df[cols['discrete']].fillna('{BLANK}')
    df[cols['continuous']] = df[cols['continuous']].fillna(0)
    print('***Done with joins, maps, ordering.')
    return (df, cols)

def build_widgets(df_source, cols, init_load=False, init_config={}, preset_options=None):
    '''
    Use a dataframe and its columns to set widget options. Widget values may
    be set by URL parameters via init_config.

    Args:
        df_source (pandas dataframe): Dataframe of the csv source.
        cols (dict): Keys are categories of columns of df_source, and values are a list of columns of that category.
        init_load (boolean, optional): If this is the initial page load, then this will be True, else False.
        init_config (dict): Initial widget configuration passed via URL.
        preset_options (list): List of strings for preset selections.

    Returns:
        wdg (ordered dict): Dictionary of bokeh.model.widgets.
    '''
    #Add widgets
    print('***Build main widgets...')
    wdg = collections.OrderedDict()
    if preset_options != None:
        wdg['presets'] = bmw.Select(title='Presets', value='None', options=['None'] + preset_options, css_classes=['wdgkey-presets'])
    wdg['x_dropdown'] = bmw.Div(text='X-Axis (required)', css_classes=['x-dropdown'])
    wdg['x'] = bmw.Select(title='X-Axis (required)', value='None', options=['None'] + cols['x-axis'], css_classes=['wdgkey-x', 'x-drop'])
    wdg['x_group'] = bmw.Select(title='Group X-Axis By', value='None', options=['None'] + cols['seriesable'], css_classes=['wdgkey-x_group', 'x-drop'])
    wdg['y_dropdown'] = bmw.Div(text='Y-Axis (required)', css_classes=['y-dropdown'])
    wdg['y'] = bmw.Select(title='Y-Axis (required)', value='None', options=['None'] + cols['y-axis'], css_classes=['wdgkey-y', 'y-drop'])
    wdg['y_agg'] = bmw.Select(title='Y-Axis Aggregation', value='Sum', options=AGGREGATIONS, css_classes=['wdgkey-y_agg', 'y-drop'])
    wdg['y_weight'] = bmw.Select(title='Weighting Factor', value='None', options=['None'] + cols['y-axis'], css_classes=['wdgkey-y_weight', 'y-drop'])
    wdg['series_dropdown'] = bmw.Div(text='Series', css_classes=['series-dropdown'])
    wdg['series'] = bmw.Select(title='Separate Series By', value='None', options=['None'] + cols['seriesable'],
        css_classes=['wdgkey-series', 'series-drop'])
    wdg['explode_dropdown'] = bmw.Div(text='Explode', css_classes=['explode-dropdown'])
    wdg['explode'] = bmw.Select(title='Explode By', value='None', options=['None'] + cols['seriesable'], css_classes=['wdgkey-explode', 'explode-drop'])
    wdg['explode_group'] = bmw.Select(title='Group Exploded Charts By', value='None', options=['None'] + cols['seriesable'],
        css_classes=['wdgkey-explode_group', 'explode-drop'])
    wdg['adv_dropdown'] = bmw.Div(text='Comparisons', css_classes=['adv-dropdown'])
    wdg['adv_op'] = bmw.Select(title='Operation', value='None', options=['None', 'Difference', 'Ratio'], css_classes=['wdgkey-adv_op', 'adv-drop'])
    wdg['adv_col'] = bmw.Select(title='Operate Across', value='None', options=['None'] + cols['all'], css_classes=['wdgkey-adv_col', 'adv-drop'])
    wdg['adv_col_base'] = bmw.Select(title='Base', value='None', options=['None'], css_classes=['wdgkey-adv_col_base', 'adv-drop'])
    wdg['filters'] = bmw.Div(text='Filters', css_classes=['filters-dropdown'])
    wdg['filters_update'] = bmw.Button(label='Update Filters', button_type='success', css_classes=['filters-update'])
    for j, col in enumerate(cols['filterable']):
        val_list = [str(i) for i in sorted(df_source[col].unique().tolist())]
        wdg['heading_filter_'+str(j)] = bmw.Div(text=col, css_classes=['filter-head'])
        wdg['filter_'+str(j)] = bmw.CheckboxGroup(labels=val_list, active=list(range(len(val_list))), css_classes=['wdgkey-filter_'+str(j), 'filter'])
    wdg['adjustments'] = bmw.Div(text='Plot Adjustments', css_classes=['adjust-dropdown'])
    wdg['chart_type'] = bmw.Select(title='Chart Type', value='Dot', options=CHARTTYPES, css_classes=['wdgkey-chart_type', 'adjust-drop'])
    wdg['plot_width'] = bmw.TextInput(title='Plot Width (px)', value=str(PLOT_WIDTH), css_classes=['wdgkey-plot_width', 'adjust-drop'])
    wdg['plot_height'] = bmw.TextInput(title='Plot Height (px)', value=str(PLOT_HEIGHT), css_classes=['wdgkey-plot_height', 'adjust-drop'])
    wdg['plot_title'] = bmw.TextInput(title='Plot Title', value='', css_classes=['wdgkey-plot_title', 'adjust-drop'])
    wdg['plot_title_size'] = bmw.TextInput(title='Plot Title Font Size', value=str(PLOT_FONT_SIZE), css_classes=['wdgkey-plot_title_size', 'adjust-drop'])
    wdg['opacity'] = bmw.TextInput(title='Opacity (0-1)', value=str(OPACITY), css_classes=['wdgkey-opacity', 'adjust-drop'])
    wdg['x_scale'] = bmw.TextInput(title='X Scale', value=str(X_SCALE), css_classes=['wdgkey-x_scale', 'adjust-drop'])
    wdg['x_min'] = bmw.TextInput(title='X Min', value='', css_classes=['wdgkey-x_min', 'adjust-drop'])
    wdg['x_max'] = bmw.TextInput(title='X Max', value='', css_classes=['wdgkey-x_max', 'adjust-drop'])
    wdg['x_title'] = bmw.TextInput(title='X Title', value='', css_classes=['wdgkey-x_title', 'adjust-drop'])
    wdg['x_title_size'] = bmw.TextInput(title='X Title Font Size', value=str(PLOT_FONT_SIZE), css_classes=['wdgkey-x_title_size', 'adjust-drop'])
    wdg['x_major_label_size'] = bmw.TextInput(title='X Labels Font Size', value=str(PLOT_AXIS_LABEL_SIZE), css_classes=['wdgkey-x_major_label_size', 'adjust-drop'])
    wdg['x_major_label_orientation'] = bmw.TextInput(title='X Labels Degrees', value=str(PLOT_LABEL_ORIENTATION),
        css_classes=['wdgkey-x_major_label_orientation', 'adjust-drop'])
    wdg['y_scale'] = bmw.TextInput(title='Y Scale', value=str(Y_SCALE), css_classes=['wdgkey-y_scale', 'adjust-drop'])
    wdg['y_min'] = bmw.TextInput(title='Y  Min', value='', css_classes=['wdgkey-y_min', 'adjust-drop'])
    wdg['y_max'] = bmw.TextInput(title='Y Max', value='', css_classes=['wdgkey-y_max', 'adjust-drop'])
    wdg['y_title'] = bmw.TextInput(title='Y Title', value='', css_classes=['wdgkey-y_title', 'adjust-drop'])
    wdg['y_title_size'] = bmw.TextInput(title='Y Title Font Size', value=str(PLOT_FONT_SIZE), css_classes=['wdgkey-y_title_size', 'adjust-drop'])
    wdg['y_major_label_size'] = bmw.TextInput(title='Y Labels Font Size', value=str(PLOT_AXIS_LABEL_SIZE), css_classes=['wdgkey-y_major_label_size', 'adjust-drop'])
    wdg['circle_size'] = bmw.TextInput(title='Circle Size (Dot Only)', value=str(CIRCLE_SIZE), css_classes=['wdgkey-circle_size', 'adjust-drop'])
    wdg['bar_width'] = bmw.TextInput(title='Bar Width (Bar Only)', value=str(BAR_WIDTH), css_classes=['wdgkey-bar_width', 'adjust-drop'])
    wdg['line_width'] = bmw.TextInput(title='Line Width (Line Only)', value=str(LINE_WIDTH), css_classes=['wdgkey-line_width', 'adjust-drop'])
    wdg['map_num'] = bmw.TextInput(title='Maps: Number of bins', value=str(MAP_NUM_BINS), css_classes=['wdgkey-map_num', 'adjust-drop'])
    wdg['map_min'] = bmw.TextInput(title='Maps: Minimum Breakpoint', value='', css_classes=['wdgkey-map_min', 'adjust-drop'])
    wdg['map_max'] = bmw.TextInput(title='Maps: Maximum Breakpoint', value='', css_classes=['wdgkey-map_max', 'adjust-drop'])
    wdg['auto_update_dropdown'] = bmw.Div(text='Auto/Manual Update', css_classes=['update-dropdown'])
    wdg['auto_update'] = bmw.Select(title='Auto Update (except filters)', value='Enable', options=['Enable', 'Disable'], css_classes=['update-drop'])
    wdg['update'] = bmw.Button(label='Manual Update', button_type='success', css_classes=['update-drop'])
    wdg['render_plots'] = bmw.Select(title='Render Plots', value='Yes', options=['Yes', 'No'], css_classes=['update-drop'])
    wdg['download_dropdown'] = bmw.Div(text='Download/Export', css_classes=['download-dropdown'])
    wdg['download'] = bmw.Button(label='Download csv of View', button_type='success', css_classes=['download-drop'])
    wdg['download_all'] = bmw.Button(label='Download csv of Source', button_type='success', css_classes=['download-drop'])
    wdg['export_config'] = bmw.Div(text='Export Config to URL', css_classes=['export-config', 'bk-bs-btn', 'bk-bs-btn-success', 'download-drop'])
    wdg['legend_dropdown'] = bmw.Div(text='Legend', css_classes=['legend-dropdown'])
    wdg['legend'] = bmw.Div(text='', css_classes=['legend-drop'])

    #use init_config (from 'widgets' parameter in URL query string) to configure widgets.
    if init_load:
        for key in init_config:
            if key in wdg:
                if hasattr(wdg[key], 'value'):
                    wdg[key].value = str(init_config[key])
                elif hasattr(wdg[key], 'active'):
                    wdg[key].active = init_config[key]

    #Add update functions for widgets
    if preset_options != None:
        wdg['presets'].on_change('value', update_reeds_presets)
    wdg['filters_update'].on_click(update_plots)
    wdg['update'].on_click(update_plots)
    wdg['download'].on_click(download)
    wdg['download_all'].on_click(download_all)
    wdg['adv_col'].on_change('value', update_adv_col)
    for name in WDG_COL:
        wdg[name].on_change('value', update_wdg_col)
    for name in WDG_NON_COL:
        wdg[name].on_change('value', update_wdg)
    print('***Done with main widgets.')
    return wdg

def set_df_plots(df_source, cols, wdg, custom_sorts={}):
    '''
    Apply filters, scaling, aggregation, and sorting to source dataframe, and return the result.

    Args:
        df_source (pandas dataframe): Dataframe of the csv source.
        cols (dict): Keys are categories of columns of df_source, and values are a list of columns of that category.
        wdg (ordered dict): Dictionary of bokeh model widgets.
        custom_sorts (dict): Keys are column names. Values are lists of values in the desired sort order.

    Returns:
        df_plots (pandas dataframe): df_source after having been filtered, scaled, aggregated, and sorted.
    '''
    print('***Filtering, Scaling, Aggregating, Adv Operations, Sorting...')
    df_plots = df_source.copy()

    #Apply filters
    for j, col in enumerate(cols['filterable']):
        active = [wdg['filter_'+str(j)].labels[i] for i in wdg['filter_'+str(j)].active]
        if col in cols['continuous']:
            active = [float(i) for i in active]
        df_plots = df_plots[df_plots[col].isin(active)]

    #Scale Axes
    if wdg['x_scale'].value != '' and wdg['x'].value in cols['continuous']:
        df_plots[wdg['x'].value] = df_plots[wdg['x'].value] * float(wdg['x_scale'].value)
    if wdg['y_scale'].value != '' and wdg['y'].value in cols['continuous']:
        df_plots[wdg['y'].value] = df_plots[wdg['y'].value] * float(wdg['y_scale'].value)

    #Apply Aggregation
    if wdg['y'].value in cols['continuous'] and wdg['y_agg'].value != 'None':
        groupby_cols = [wdg['x'].value]
        if wdg['x_group'].value != 'None': groupby_cols = [wdg['x_group'].value] + groupby_cols
        if wdg['series'].value != 'None': groupby_cols = [wdg['series'].value] + groupby_cols
        if wdg['explode'].value != 'None': groupby_cols = [wdg['explode'].value] + groupby_cols
        if wdg['explode_group'].value != 'None': groupby_cols = [wdg['explode_group'].value] + groupby_cols
        df_grouped = df_plots.groupby(groupby_cols, sort=False)
        if wdg['y_agg'].value == 'Sum':
            df_plots = df_grouped[wdg['y'].value].sum().reset_index()
        elif wdg['y_agg'].value == 'Ave':
            df_plots = df_grouped[wdg['y'].value].mean().reset_index()
        elif wdg['y_agg'].value == 'Weighted Ave' and wdg['y_weight'].value in cols['continuous']:
            df_plots = df_grouped.apply(wavg, wdg['y'].value, wdg['y_weight'].value).reset_index()
            df_plots.rename(columns={0: wdg['y'].value}, inplace=True)

    #Do Advanced Operations
    op = wdg['adv_op'].value
    col = wdg['adv_col'].value
    col_base = wdg['adv_col_base'].value
    y_val = wdg['y'].value
    y_agg = wdg['y_agg'].value
    if op != 'None' and col != 'None' and col in df_plots and col_base != 'None' and y_agg != 'None' and y_val in cols['continuous']:
        #sort df_plots so that col_base is at the front, so that we can use transform('first') later
        if col in cols['continuous'] and col_base not in ADV_BASES:
            col_base = float(col_base)
        col_list = df_plots[col].unique().tolist()
        if col_base not in ADV_BASES:
            col_list.remove(col_base)
            col_list = [col_base] + col_list
        df_plots['tempsort'] = df_plots[col].map(lambda x: col_list.index(x))
        df_plots = df_plots.sort_values('tempsort').reset_index(drop=True)
        df_plots.drop(['tempsort'], axis='columns', inplace=True)
        #groupby all columns that are not the operating column and y axis column so we can do operations on y-axis across the operating column
        groupcols = [i for i in df_plots.columns.values.tolist() if i not in [col, y_val]]
        if groupcols != []:
            df_grouped = df_plots.groupby(groupcols, sort=False)[y_val]
        else:
            #if we don't have other columns to group, make one, to prevent error
            df_plots['tempgroup'] = 1
            df_grouped = df_plots.groupby('tempgroup', sort=False)[y_val]
        #Now do operations with the groups:
        if op == 'Difference':
            if col_base == 'Consecutive':
                df_plots[y_val] = df_grouped.diff()
            elif col_base == 'Total':
                df_plots[y_val] = df_plots[y_val] - df_grouped.transform('sum')
            else:
                df_plots[y_val] = df_plots[y_val] - df_grouped.transform('first')
        elif op == 'Ratio':
            if col_base == 'Consecutive':
                df_plots[y_val] = df_grouped.diff()
            elif col_base == 'Total':
                df_plots[y_val] = df_plots[y_val] / df_grouped.transform('sum')
            else:
                df_plots[y_val] = df_plots[y_val] / df_grouped.transform('first')
        #Finally, clean up df_plots, dropping unnecessary columns, rows with the base value, and any rows with NAs for y_vals
        if 'tempgroup' in df_plots:
            df_plots.drop(['tempgroup'], axis='columns', inplace=True)
        df_plots = df_plots[~df_plots[col].isin([col_base])]
        df_plots = df_plots[pd.notnull(df_plots[y_val])]

    #Sort Dataframe
    sortby_cols = [wdg['x'].value]
    if wdg['x_group'].value != 'None': sortby_cols = [wdg['x_group'].value] + sortby_cols
    if wdg['series'].value != 'None': sortby_cols = [wdg['series'].value] + sortby_cols
    if wdg['explode'].value != 'None': sortby_cols = [wdg['explode'].value] + sortby_cols
    if wdg['explode_group'].value != 'None': sortby_cols = [wdg['explode_group'].value] + sortby_cols
    #Add custom sort columns
    temp_sort_cols = sortby_cols[:]
    for col in custom_sorts:
        if col in sortby_cols:
            df_plots[col + '__sort_col'] = df_plots[col].map(lambda x: custom_sorts[col].index(x))
            temp_sort_cols[sortby_cols.index(col)] = col + '__sort_col'
    #Do sorting
    df_plots = df_plots.sort_values(temp_sort_cols).reset_index(drop=True)
    #Remove custom sort columns
    for col in custom_sorts:
        if col in sortby_cols:
            df_plots = df_plots.drop(col + '__sort_col', 1)

    #Rearrange column order for csv download
    unsorted_columns = [col for col in df_plots.columns if col not in sortby_cols + [wdg['y'].value]]
    df_plots = df_plots[unsorted_columns + sortby_cols + [wdg['y'].value]]
    print('***Done Filtering, Scaling, Aggregating, Adv Operations, Sorting.')
    return df_plots

def create_figures(df_plots, wdg, cols):
    '''
    Create figures based on the data in a dataframe and widget configuration, and return figures in a list.
    The explode widget determines if there will be multiple figures.

    Args:
        df_plots (pandas dataframe): Dataframe of csv source after being filtered, scaled, aggregated, and sorted.
        wdg (ordered dict): Dictionary of bokeh model widgets.
        cols (dict): Keys are categories of columns of df_source, and values are a list of columns of that category.

    Returns:
        plot_list (list): List of bokeh.model.figures.
    '''
    print('***Building Figures...')
    plot_list = []
    df_plots_cp = df_plots.copy()
    if wdg['explode'].value == 'None':
        plot_list.append(create_figure(df_plots_cp, df_plots, wdg, cols))
    else:
        if wdg['explode_group'].value == 'None':
            for explode_val in df_plots_cp[wdg['explode'].value].unique().tolist():
                df_exploded = df_plots_cp[df_plots_cp[wdg['explode'].value].isin([explode_val])]
                plot_list.append(create_figure(df_exploded, df_plots, wdg, cols, explode_val))
        else:
            for explode_group in df_plots_cp[wdg['explode_group'].value].unique().tolist():
                df_exploded_group = df_plots_cp[df_plots_cp[wdg['explode_group'].value].isin([explode_group])]
                for explode_val in df_exploded_group[wdg['explode'].value].unique().tolist():
                    df_exploded = df_exploded_group[df_exploded_group[wdg['explode'].value].isin([explode_val])]
                    plot_list.append(create_figure(df_exploded, df_plots, wdg, cols, explode_val, explode_group))
    print('***Done Building Figures.')
    return plot_list

def create_figure(df_exploded, df_plots, wdg, cols, explode_val=None, explode_group=None):
    '''
    Create and return a figure based on the data in a dataframe and widget configuration.

    Args:
        df_exploded (pandas dataframe): Dataframe of just the data that will be plotted in this figure.
        df_plots (pandas dataframe): Dataframe of all plots data, used only for maintaining consistent series colors.
        wdg (ordered dict): Dictionary of bokeh model widgets.
        cols (dict): Keys are categories of columns of df_source, and values are a list of columns of that category.
        explode_val (string, optional): The value in the column designated by wdg['explode'] that applies to this figure.
        explode_group (string, optional): The value in the wdg['explode_group'] column that applies to this figure.

    Returns:
        p (bokeh.model.figure): A figure, with all glyphs added by the add_glyph() function.
    '''
    # If x_group has a value, create a combined column in the dataframe for x and x_group
    x_col = wdg['x'].value
    if wdg['x_group'].value != 'None':
        x_col = str(wdg['x_group'].value) + '_' + str(wdg['x'].value)
        df_exploded[x_col] = df_exploded[wdg['x_group'].value].map(str) + ' ' + df_exploded[wdg['x'].value].map(str)

    #Build x and y ranges and figure title
    kw = dict()

    #Set x and y ranges. When x is grouped, there is added complication of separating the groups
    xs = df_exploded[x_col].values.tolist()
    ys = df_exploded[wdg['y'].value].values.tolist()
    if wdg['x_group'].value != 'None':
        kw['x_range'] = []
        unique_groups = df_exploded[wdg['x_group'].value].unique().tolist()
        unique_xs = df_exploded[wdg['x'].value].unique().tolist()
        for i, ugr in enumerate(unique_groups):
            for uxs in unique_xs:
                kw['x_range'].append(str(ugr) + ' ' + str(uxs))
            #Between groups, add entries that consist of spaces. Increase number of spaces from
            #one break to the next so that each entry is unique
            kw['x_range'].append(' ' * (i + 1))
    elif wdg['x'].value in cols['discrete']:
        kw['x_range'] = sorted(set(xs))
    if wdg['y'].value in cols['discrete']:
        kw['y_range'] = sorted(set(ys))

    #Set figure title
    kw['title'] = wdg['plot_title'].value
    seperator = '' if kw['title'] == '' else ', '
    if explode_val is not None:
        if explode_group is not None:
            kw['title'] = kw['title'] + seperator + "%s = %s" % (wdg['explode_group'].value, str(explode_group))
        seperator = '' if kw['title'] == '' else ', '
        kw['title'] = kw['title'] + seperator + "%s = %s" % (wdg['explode'].value, str(explode_val))

    #Add figure tools
    hover = bmt.HoverTool(
            tooltips=[
                ("ser", "@ser_legend"),
                ("x", "@x_legend"),
                ("y", "@y_legend"),
            ]
    )
    TOOLS = [bmt.PanTool(), bmt.WheelZoomTool(), hover, bmt.ResetTool(), bmt.SaveTool()]

    #Create figure with the ranges, titles, and tools, and adjust formatting and labels
    p = bp.figure(plot_height=int(wdg['plot_height'].value), plot_width=int(wdg['plot_width'].value), tools=TOOLS, **kw)
    p.toolbar.active_drag = TOOLS[0]
    p.title.text_font_size = wdg['plot_title_size'].value + 'pt'
    p.xaxis.axis_label = wdg['x_title'].value
    p.yaxis.axis_label = wdg['y_title'].value
    p.xaxis.axis_label_text_font_size = wdg['x_title_size'].value + 'pt'
    p.yaxis.axis_label_text_font_size = wdg['y_title_size'].value + 'pt'
    p.xaxis.major_label_text_font_size = wdg['x_major_label_size'].value + 'pt'
    p.yaxis.major_label_text_font_size = wdg['y_major_label_size'].value + 'pt'
    p.xaxis.major_label_orientation = 'horizontal' if wdg['x_major_label_orientation'].value == '0' else math.radians(float(wdg['x_major_label_orientation'].value))
    if wdg['x'].value in cols['continuous']:
        if wdg['x_min'].value != '': p.x_range.start = float(wdg['x_min'].value)
        if wdg['x_max'].value != '': p.x_range.end = float(wdg['x_max'].value)
    if wdg['y'].value in cols['continuous']:
        if wdg['y_min'].value != '': p.y_range.start = float(wdg['y_min'].value)
        if wdg['y_max'].value != '': p.y_range.end = float(wdg['y_max'].value)

    #Add glyphs to figure
    c = C_NORM
    if wdg['series'].value == 'None':
        if wdg['y_agg'].value != 'None' and wdg['y'].value in cols['continuous']:
            xs = df_exploded[x_col].values.tolist()
            ys = df_exploded[wdg['y'].value].values.tolist()
        add_glyph(wdg, p, xs, ys, c)
    else:
        full_series = df_plots[wdg['series'].value].unique().tolist() #for colors only
        if wdg['chart_type'].value in STACKEDTYPES: #We are stacking the series
            xs_full = sorted(df_exploded[x_col].unique().tolist())
            y_bases_pos = [0]*len(xs_full)
            y_bases_neg = [0]*len(xs_full)
        for i, ser in enumerate(df_exploded[wdg['series'].value].unique().tolist()):
            c = COLORS[full_series.index(ser)]
            df_series = df_exploded[df_exploded[wdg['series'].value].isin([ser])]
            xs_ser = df_series[x_col].values.tolist()
            ys_ser = df_series[wdg['y'].value].values.tolist()
            if wdg['chart_type'].value not in STACKEDTYPES: #The series will not be stacked
                add_glyph(wdg, p, xs_ser, ys_ser, c, series=ser)
            else: #We are stacking the series
                ys_pos = [ys_ser[xs_ser.index(x)] if x in xs_ser and ys_ser[xs_ser.index(x)] > 0 else 0 for i, x in enumerate(xs_full)]
                ys_neg = [ys_ser[xs_ser.index(x)] if x in xs_ser and ys_ser[xs_ser.index(x)] < 0 else 0 for i, x in enumerate(xs_full)]
                ys_stacked_pos = [ys_pos[i] + y_bases_pos[i] for i in range(len(xs_full))]
                ys_stacked_neg = [ys_neg[i] + y_bases_neg[i] for i in range(len(xs_full))]
                add_glyph(wdg, p, xs_full, ys_stacked_pos, c, y_bases=y_bases_pos, series=ser)
                add_glyph(wdg, p, xs_full, ys_stacked_neg, c, y_bases=y_bases_neg, series=ser)
                y_bases_pos = ys_stacked_pos
                y_bases_neg = ys_stacked_neg
    return p

def add_glyph(wdg, p, xs, ys, c, y_bases=None, series=None):
    '''
    Add a glyph to a Bokeh figure, depending on the chosen chart type.

    Args:
        wdg (ordered dict): Dictionary of bokeh model widgets.
        p (bokeh.model.figure): Bokeh figure.
        xs (list): List of x-values. These could be numeric or strings.
        ys (list): List of y-values. These could be numeric or strings. If series data is stacked, these values include stacking.
        c (string): Color to use for this series.
        y_bases (list, optional): Only used when stacking series. This is the previous cumulative stacking level.
        series (string): Name of current series for this glyph.

    Returns:
        Nothing.
    '''
    alpha = float(wdg['opacity'].value)
    y_unstacked = list(ys) if y_bases is None else [ys[i] - y_bases[i] for i in range(len(ys))]
    ser = ['None']*len(xs) if series is None else [series]*len(xs)
    if wdg['chart_type'].value == 'Dot':
        source = bms.ColumnDataSource({'x': xs, 'y': ys, 'x_legend': xs, 'y_legend': y_unstacked, 'ser_legend': ser})
        p.circle('x', 'y', source=source, color=c, size=int(wdg['circle_size'].value), fill_alpha=alpha, line_color=None, line_width=None)
    elif wdg['chart_type'].value == 'Line':
        source = bms.ColumnDataSource({'x': xs, 'y': ys, 'x_legend': xs, 'y_legend': y_unstacked, 'ser_legend': ser})
        p.line('x', 'y', source=source, color=c, alpha=alpha, line_width=float(wdg['line_width'].value))
    elif wdg['chart_type'].value == 'Bar' and y_unstacked != [0]*len(y_unstacked):
        if y_bases is None: y_bases = [0]*len(ys)
        centers = [(ys[i] + y_bases[i])/2 for i in range(len(ys))]
        heights = [abs(ys[i] - y_bases[i]) for i in range(len(ys))]
        #bars have issues when height is 0, so remove elements whose height is 0 
        heights_orig = list(heights) #we make a copy so we aren't modifying the list we are iterating on.
        xs_cp = list(xs) #we don't want to modify xs that are passed into function
        for i, h in reversed(list(enumerate(heights_orig))):
            if h == 0:
                del xs_cp[i]
                del centers[i]
                del heights[i]
                del y_unstacked[i]
                del ser[i]
        source = bms.ColumnDataSource({'x': xs_cp, 'y': centers, 'x_legend': xs_cp, 'y_legend': y_unstacked, 'h': heights, 'ser_legend': ser})
        p.rect('x', 'y', source=source, height='h', color=c, fill_alpha=alpha, width=float(wdg['bar_width'].value), line_color=None, line_width=None)
    elif wdg['chart_type'].value == 'Area' and y_unstacked != [0]*len(y_unstacked):
        if y_bases is None: y_bases = [0]*len(ys)
        xs_around = xs + xs[::-1]
        ys_around = y_bases + ys[::-1]
        source = bms.ColumnDataSource({'x': [xs_around], 'y': [ys_around], 'ser_legend': [series]})
        p.patches('x', 'y', source=source, alpha=alpha, fill_color=c, line_color=None, line_width=None)

def create_maps(df, wdg, cols):
    '''
    Create maps
    '''
    maps = []
    legend_labels = []
    regions = ['i','n','r','rnew','rto','st']
    x_axis = df.iloc[:,-2]
    y_axis = df.iloc[:,-1]
    if x_axis.name not in regions or y_axis.dtype == object:
        return (maps, legend_labels) #empty list
    #find x and y ranges based on the mins and maxes of the regional boundaries for only regions that
    #are in the data
    filepath = this_dir_path + '/csv/gis_' + x_axis.name + '.csv'
    region_boundaries = pd.read_csv(filepath, sep=',')
    #load hierarchy.csv and join on region_boundaries
    df_join = pd.read_csv(this_dir_path + '/csv/hierarchy.csv', sep=',')
    #remove columns to left of x_axis.name in df_join
    for c in df_join.columns.values.tolist():
        if c == x_axis.name:
            break
        df_join.drop(c, axis=1, inplace=True)
    #remove duplicate rows
    df_join.drop_duplicates(subset=x_axis.name, inplace=True)
    #merge df_join into df
    region_boundaries = pd.merge(left=region_boundaries, right=df_join, left_on='id', right_on=x_axis.name, sort=False)
    #filter region_boundaries by filter widgets
    for j, col in enumerate(cols['filterable']):
        if col in region_boundaries:
            active = [wdg['filter_'+str(j)].labels[i] for i in wdg['filter_'+str(j)].active]
            region_boundaries = region_boundaries[region_boundaries[col].isin(active)]
    #Add x and y columns to region_boundaries and find x and y ranges
    region_boundaries['x'] = region_boundaries['long']*53
    region_boundaries['y'] = region_boundaries['lat']*69
    ranges = {
        'x_max': region_boundaries['x'].max(),
        'x_min': region_boundaries['x'].min(),
        'y_max': region_boundaries['y'].max(),
        'y_min': region_boundaries['y'].min(),
    }
    #determine bins of values
    map_num_bins = int(wdg['map_num'].value) if wdg['map_num'].value != '' else 5
    if wdg['map_min'].value != '' and wdg['map_max'].value != '':
        map_min = float(wdg['map_min'].value)
        map_max = float(wdg['map_max'].value)
        bin_width = (map_max - map_min)/(map_num_bins - 2)
        breakpoints = [map_min + bin_width*i for i in range(map_num_bins - 1)]
        breakpoint_strings = [str(bp) for bp in breakpoints]
    else:
        bin_width = float(y_axis.max() - y_axis.min())/map_num_bins
        map_min = y_axis.min() + bin_width
        map_max = y_axis.max() - bin_width
        breakpoints = [map_min + bin_width*i for i in range(map_num_bins - 1)]
        breakpoint_strings = ['%.2E' % bp for bp in breakpoints]
    #gather legend_labels array
    legend_labels = ['< ' + breakpoint_strings[0]]
    legend_labels += [breakpoint_strings[i] + ' - ' + breakpoint_strings[i+1] for i in range(map_num_bins - 2)]
    legend_labels += ['> ' + breakpoint_strings[-1]]

    df_maps = df.copy()
    #assign all y-values to bins
    df_maps['bin_index'] = y_axis.apply(get_map_bin_index, args=(breakpoints,))
    #If there are only 3 columns (x_axis, y_axis, and bin_index), that means we aren't exploding:
    if len(df_maps.columns) == 3:
        maps.append(create_map(df_maps, ranges, region_boundaries, wdg))
        return (maps, legend_labels) #single map
    #Otherwise we are exploding.
    #find all unique groups of the explode columns.
    df_unique = df_maps.copy()
    #remove x, y, and bin_index
    df_unique = df_unique[df_unique.columns[0:-3]]
    df_unique.drop_duplicates(inplace=True)
    #Loop through rows of df_unique, filter df_maps based on values in each row,
    #and send filtered dataframe to mapping function
    for i, row in df_unique.iterrows():
        df_map = df_maps
        title = ''
        for col in df_unique:
            df_map = df_map[df_map[col] == row[col]]
            title = title + col + '=' + row[col] + ', '
        #preserve just x axis, y axis, and bin index
        df_map = df_map[df_map.columns[-3:]]
        #remove final comma of title
        title = title[:-2]
        maps.append(create_map(df_map, ranges, region_boundaries, wdg, title))
    return (maps, legend_labels) #multiple maps

def get_map_bin_index(val, breakpoints):
    for i, breakpoint in enumerate(breakpoints):
        if val < breakpoint:
            return i
    return len(breakpoints)

def create_map(df, ranges, region_boundaries, wdg, title=''):
    '''
    Create map
    '''

    df_regions = df.iloc[:,0].tolist()
    df_values = df.iloc[:,1].tolist()
    df_bins = df.iloc[:,2].tolist()

    xs = [] #list of lists of x values of boundaries of regions
    ys = [] #list of lists of y values of boundaries of regions
    regions = []
    values = []
    colors = []
    for reg in region_boundaries['id'].unique().tolist():
        region_boundary = region_boundaries[region_boundaries['id'] == reg]
        xs.append(region_boundary['x'].values.tolist())
        ys.append(region_boundary['y'].values.tolist())
        regions.append(reg)
        if reg in df_regions:
            index = df_regions.index(reg)
            value = df_values[index]
            values.append(value)
            bin_num = df_bins[index]
            colors.append(COLORS[int(bin_num)])
        else:
            values.append('NA')
            colors.append('#ffffff')

    source = bms.ColumnDataSource(data=dict(
        x=xs,
        y=ys,
        region=regions,
        value=values,
        color=colors,
    ))
    #Add figure tools
    hover = bmt.HoverTool(
            tooltips=[
                ("reg", "@region"),
                ("val", "@value"),
            ],
            point_policy = "follow_mouse",
    )
    TOOLS = [bmt.PanTool(), bmt.WheelZoomTool(), hover, bmt.ResetTool(), bmt.SaveTool()]
    #find max and min of xs and ys to set aspect ration of map
    
    aspect_ratio = (ranges['y_max'] - ranges['y_min'])/(ranges['x_max'] - ranges['x_min'])
    width = wdg['plot_width'].value
    height = aspect_ratio * float(width)
    fig_map = bp.figure(
        title=title,
        plot_height=int(height),
        plot_width=int(width),
        x_range=(ranges['x_min'], ranges['x_max']),
        y_range=(ranges['y_min'], ranges['y_max']),
        x_axis_location=None,
        y_axis_location=None,
        tools=TOOLS
    )
    fig_map.title.text_font_size = wdg['plot_title_size'].value + 'pt'
    fig_map.grid.grid_line_color = None
    fig_map.patches('x', 'y', source=source, fill_color='color', fill_alpha=float(wdg['opacity'].value), line_color="black", line_width=0.5)
    return fig_map

def build_map_legend(labels):
    '''
    Return html for series legend, based on values of column that was chosen for series, and global COLORS.

    Args:

    Returns:
        legend_string (string): html to be used as legend.
    '''
    colors = [COLORS[i] for i, t in enumerate(labels)]
    legend_string = build_legend(labels, colors)
    return legend_string

def build_plot_legend(df_plots, series_val):
    '''
    Return html for series legend, based on values of column that was chosen for series, and global COLORS.

    Args:
        df_plots (pandas dataframe): Dataframe of all plots data.
        series_val (string): Header for column chosen as series.

    Returns:
        legend_string (string): html to be used as legend.
    '''
    if series_val == 'None':
        return ''
    labels = df_plots[series_val].unique().tolist()
    colors = [COLORS[i] for i, t in enumerate(labels)]
    labels.reverse()
    colors.reverse()
    legend_string = build_legend(labels, colors)
    return legend_string

def build_legend(labels, colors):
    '''
    Return html for legend

    Args:

    Returns:
        legend_string (string): html to be used as legend.
    '''
    legend_string = ''
    for i, txt in enumerate(labels):
        legend_string += '<div class="legend-entry"><span class="legend-color" style="background-color:' + str(colors[i]) + ';"></span>'
        legend_string += '<span class="legend-text">' + str(txt) +'</span></div>'
    return legend_string

def wavg(group, avg_name, weight_name):
    """ http://pbpython.com/weighted-average.html
    """
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return 0


def update_data(attr, old, new):
    '''
    When data source is updated
    '''
    update_data_source()

def update_data_source(init_load=False, init_config={}):
    GL['widgets'] = GL['data_source_wdg'].copy()
    path = GL['data_source_wdg']['data'].value
    path = path.replace('"', '')
    if path == '':
        pass
    elif path.lower().endswith('.csv'):
        GL['df_source'], GL['columns'] = get_df_csv(path)
        GL['widgets'].update(build_widgets(GL['df_source'], GL['columns'], init_load, init_config))
    elif path.lower().endswith('.gdx'):
        GL['widgets'].update(get_wdg_gdx(path, GL['widgets']))
    else: #reeds
        GL['variant_wdg'] = get_wdg_reeds(path, init_load, init_config)
        GL['widgets'].update(GL['variant_wdg'])
        #if this is the initial load, we need to build the rest of the widgets if we've selected a result.
        if init_load and GL['variant_wdg']['result'].value is not 'None':
            get_reeds_data(GL['variant_wdg'])
            GL['df_source'], GL['columns'] = process_reeds_data(GL['variant_wdg'])
            preset_options = []
            if 'presets' in results_meta[GL['variant_wdg']['result'].value]:
                preset_options = results_meta[GL['variant_wdg']['result'].value]['presets'].keys()
            GL['widgets'].update(build_widgets(GL['df_source'], GL['columns'], init_load, init_config, preset_options))
    GL['controls'].children = list(GL['widgets'].values())
    GL['plots'].children = []

def update_reeds_meta(attr, old, new):
    update_reeds_wdg(type='meta')

def update_reeds_result(attr, old, new):
    update_reeds_wdg(type='result')

def update_reeds_wdg(type):
    if 'result' in GL['variant_wdg'] and GL['variant_wdg']['result'].value is not 'None':
        preset_options = []
        if type == 'result':
            get_reeds_data(GL['variant_wdg'])
            if 'presets' in results_meta[GL['variant_wdg']['result'].value]:
                preset_options = results_meta[GL['variant_wdg']['result'].value]['presets'].keys()
        GL['df_source'], GL['columns'] = process_reeds_data(GL['variant_wdg'])
        GL['widgets'].update(build_widgets(GL['df_source'], GL['columns'], preset_options=preset_options))
    GL['controls'].children = list(GL['widgets'].values())
    update_plots()

def update_reeds_presets(attr, old, new):
    wdg = GL['widgets']
    if wdg['presets'].value != 'None':
        #set x to "None" so that the chart will not render until all presets have been set.
        wdg['x'].value = 'None'
        preset = results_meta[wdg['result'].value]['presets'][wdg['presets'].value]
        preset_keys_minus_x = [key for key in preset if key != 'x']
        for key in preset_keys_minus_x:
            wdg[key].value = preset[key]
        wdg['x'].value = preset['x']

def update_wdg(attr, old, new):
    '''
    When general widgets are updated (not in WDG_COL), update plots only.
    '''
    if GL['widgets']['auto_update'].value == 'Enable':
        update_plots()

def update_wdg_col(attr, old, new):
    '''
    When widgets in WDG_COL are updated, set the options of all WDG_COL widgets,
    and update plots.
    '''
    set_wdg_col_options()
    if GL['widgets']['auto_update'].value == 'Enable':
        update_plots()

def update_adv_col(attr, old, new):
    '''
    When adv_col is set, find unique values of adv_col in dataframe, and set adv_col_base with those values.
    '''
    wdg = GL['widgets']
    df = GL['df_source']
    if wdg['adv_col'].value != 'None':
        wdg['adv_col_base'].options = ['None'] + ADV_BASES + [str(i) for i in sorted(df[wdg['adv_col'].value].unique().tolist())]

def set_wdg_col_options():
    '''
    Limit available options for WDG_COL widgets based on their selected values, so that users
    cannot select the same value for two different WDG_COL widgets.
    '''
    cols = GL['columns']
    wdg = GL['widgets']
    #get list of selected values and use to reduce selection options.
    sels = [str(wdg[w].value) for w in WDG_COL if str(wdg[w].value) !='None']
    for w in WDG_COL:
        val = str(wdg[w].value)
        none_append = [] if val == 'None' else ['None']
        if w == 'x':
            opt_reduced = [x for x in cols['x-axis'] if x not in sels]
        elif w == 'y':
            opt_reduced = [x for x in cols['y-axis'] if x not in sels]
        else:
            opt_reduced = [x for x in cols['seriesable'] if x not in sels]
        wdg[w].options = [val] + opt_reduced + none_append

def update_plots():
    '''
    Make sure x axis and y axis are set. If so, set the dataframe for the plots and build them.
    '''
    if GL['widgets']['x'].value == 'None' or GL['widgets']['y'].value == 'None':
        GL['plots'].children = []
        return
    GL['df_plots'] = set_df_plots(GL['df_source'], GL['columns'], GL['widgets'], custom_sorts)
    if GL['widgets']['render_plots'].value == 'Yes':
        if GL['widgets']['chart_type'].value == 'Map':
            figs, legend_labels = create_maps(GL['df_plots'], GL['widgets'], GL['columns'])
            legend_text = build_map_legend(legend_labels)
        else:
            figs = create_figures(GL['df_plots'], GL['widgets'], GL['columns'])
            legend_text = build_plot_legend(GL['df_plots'], GL['widgets']['series'].value)
        GL['widgets']['legend'].text = legend_text
        GL['plots'].children = figs

def download():
    '''
    Download a csv file of the currently viewed data to the downloads/ directory,
    with the current timestamp.
    '''
    print('***Downloading View...')
    path = this_dir_path + '/downloads/out '+ datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S-%f")+'.csv'
    GL['df_plots'].to_csv(path, index=False)
    print('***Done downloading View to ' + path)

def download_all():
    '''
    Download a csv file of the full data source to the downloads/ directory,
    with the current timestamp.
    '''
    print('***Downloading full source...')
    path = this_dir_path + '/downloads/out '+ datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S-%f")+'.csv'
    GL['df_source'].to_csv(path, index=False)
    print('***Done downloading full source to ' + path)

#This sets the whole thing in motion when a page is loaded
initialize()
