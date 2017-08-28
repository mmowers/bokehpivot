'''
ReEDS functions and globals for bokehpivot integration
'''
from __future__ import division
import os
import copy
import pandas as pd
import collections
import bokeh.models.widgets as bmw
import gdx2py
import reeds
import core

this_dir_path = os.path.dirname(os.path.realpath(__file__))

#ReEDS globals
#scenarios: each element is a dict with name of scenario and path to scenario
#result_dfs: keys are ReEDS result names. Values are dataframes for that result (with 'scenario' as one of the columns)
GL_REEDS = {'scenarios': [], 'result_dfs': {}}

def reeds_static(data_source, static_presets, base=None, report_format='both'):
    '''
    Build static html and excel reports based on specified ReEDS presets
    Args:
        data_source (string): Path to ReEDS runs that will be included in report
        static_presets (list of dicts): List of presets for which to make report. Each preset has these keys:
            'name' (required): Displayed name of the result 
            'result': ReEDS result name in reeds.py.
            'preset': ReEDS result preset as defined in reeds.py
            'modify': Preset modifications, either 'base_only' or 'diff'.
            'config': Custom widget configuration. This configuration will be added on top of 'result', 'preset', 'modify', if they are present.
        base (string): Identifier for base scenario, if making comparison charts
        report_format (string): 'html', 'excel', or 'both', specifying which reports to make
    Returns:
        Nothing: HTML and Excel files are created
    '''
    core_presets = []
    for static_preset in static_presets:
        #build the full widget configuration for each preset.
        config = {}
        if 'result' in static_preset:
            config.update({'result': static_preset['result']})
            if 'preset' in static_preset:
                #deepcopy is needed to prevent 'filter', a key within the preset, from pointing to the same object in config as in the original preset dict in results_meta.
                preset_config = copy.deepcopy(reeds.results_meta[static_preset['result']]['presets'][static_preset['preset']])
                config.update(preset_config)
        if 'modify' in static_preset:
            if static_preset['modify'] == 'base_only':
                #if designated as base_only, filter to only include base scenario
                if 'filter' not in config:
                    config['filter'] = {}
                config['filter'].update({'scenario': [base]})
            elif static_preset['modify'] == 'diff':
                #find differences with base. First set x to 'None' to prevent updating, then reset x at the end of the widget updates.
                config.update({'adv_op': 'Difference', 'adv_col': 'scenario', 'adv_col_base': base})
        if 'config' in static_preset:
            config.update(static_preset['config'])
        core_presets.append({'name': static_preset['name'], 'config': config})
    core.static_report(data_source, core_presets, report_format)

def get_wdg_reeds(path, init_load, wdg_config, wdg_defaults, custom_sorts):
    '''
    From data source path, fetch paths to scenarios and return dict of widgets for
    meta files, scenarios, and results

    Args:
        path (string): Path to a ReEDS run folder or a folder containing ReEDS runs folders.
        init_load (Boolean): True if this is the initial page load. False otherwise.
        wdg_config (dict): Initial configuration for widgets.
        wdg_defaults (dict): Keys are widget names and values are the default values of the widgets.
        custom_sorts (dict): Keys are column names. Values are lists of values in the desired sort order.

    Returns:
        topwdg (ordered dict): Dictionary of bokeh.model.widgets.
        scenarios (array of dicts): Each element is a dict with name of scenario and path to scenario.
    '''
    print('***Fetching ReEDS scenarios...')
    topwdg = collections.OrderedDict()

    #Meta widgets
    topwdg['meta'] = bmw.Div(text='Meta', css_classes=['meta-dropdown'])
    for col in reeds.columns_meta:
        if 'map' in reeds.columns_meta[col]:
            topwdg['meta_map_'+col] = bmw.TextInput(title='"'+col+ '" Map', value=reeds.columns_meta[col]['map'], css_classes=['wdgkey-meta_map_'+col, 'meta-drop'])
        if 'join' in reeds.columns_meta[col]:
            topwdg['meta_join_'+col] = bmw.TextInput(title='"'+col+ '" Join', value=reeds.columns_meta[col]['join'], css_classes=['wdgkey-meta_join_'+col, 'meta-drop'])
        if 'style' in reeds.columns_meta[col]:
            topwdg['meta_style_'+col] = bmw.TextInput(title='"'+col+ '" Style', value=reeds.columns_meta[col]['style'], css_classes=['wdgkey-meta_style_'+col, 'meta-drop'])

    #Filter Scenarios widgets and Result widget
    scenarios = []
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
                    if os.path.isfile(abs_path_scen+'/gdxfiles/CONVqn.gdx'):
                        custom_sorts['scenario'].append(scen['name'])
                        scenarios.append({'name': scen['name'], 'path': abs_path_scen})
        #Else if the path is pointing to a directory, check if the directory is a run folder
        #containing gdxfiles/ and use this as the lone scenario. Otherwise, it must contain
        #run folders, so gather all of those scenarios.
        elif os.path.isdir(runs_path):
            abs_path = str(os.path.abspath(runs_path))
            if os.path.isfile(abs_path+'/gdxfiles/CONVqn.gdx'):
                scenarios.append({'name': os.path.basename(abs_path), 'path': abs_path})
            else:
                subdirs = os.walk(abs_path).next()[1]
                for subdir in subdirs:
                    if os.path.isfile(abs_path+'/'+subdir+'/gdxfiles/CONVqn.gdx'):
                        abs_subdir = str(os.path.abspath(abs_path+'/'+subdir))
                        scenarios.append({'name': subdir, 'path': abs_subdir})
    #If we have scenarios, build widgets for scenario filters and result.
    for key in ["scenario_filter_dropdown", "scenario_filter", "result"]:
        topwdg.pop(key, None)
    if scenarios:
        labels = [a['name'] for a in scenarios]
        topwdg['scenario_filter_dropdown'] = bmw.Div(text='Filter Scenarios', css_classes=['scenario-filter-dropdown'])
        topwdg['scenario_filter'] = bmw.CheckboxGroup(labels=labels, active=list(range(len(labels))), css_classes=['wdgkey-scenario_filter'])
        topwdg['result'] = bmw.Select(title='Result', value='None', options=['None']+list(reeds.results_meta.keys()), css_classes=['wdgkey-result'])
    #save defaults
    core.save_wdg_defaults(topwdg, wdg_defaults)
    #set initial config
    if init_load:
        core.initialize_wdg(topwdg, wdg_config)
    #Add update functions
    for key in topwdg:
        if key.startswith('meta_'):
            topwdg[key].on_change('value', update_reeds_meta)
    topwdg['result'].on_change('value', update_reeds_result)
    
    print('***Done fetching ReEDS scenarios.')
    return (topwdg, scenarios)

def get_reeds_data(topwdg, scenarios, result_dfs):
    '''
    For a selected ReEDS result and set of scenarios, fetch gdx data,
    preprocess it, and add to global result_dfs dictionary if the data
    hasn't already been fetched.

    Args:
        topwdg (ordered dict): ReEDS widgets (meta widgets, scenarios widget, result widget)
        scenarios (array of dicts): Each element is a dict with name of scenario and path to scenario.
        result_dfs (dict): Keys are ReEDS result names. Values are dataframes for that result (with 'scenario' as one of the columns)

    Returns:
        Nothing: result_dfs is modified
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
    result_meta = reeds.results_meta[result]
    for i in topwdg['scenario_filter'].active:
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
            #preprocess and return one dataframe
            if 'preprocess' in result_meta:
                for preprocess in result_meta['preprocess']:
                    df_scen_result = preprocess['func'](df_scen_result, **preprocess['args'])
            #preprocess columns in this dataframe
            for col in df_scen_result.columns.values.tolist():
                if col in reeds.columns_meta and 'preprocess' in reeds.columns_meta[col]:
                    for preprocess in reeds.columns_meta[col]['preprocess']:
                        df_scen_result[col] = preprocess(df_scen_result[col])
            df_scen_result['scenario'] = scenario_name
            if result_dfs[result] is None:
                result_dfs[result] = df_scen_result
            else:
                result_dfs[result] = pd.concat([result_dfs[result], df_scen_result]).reset_index(drop=True)
        print('***Done fetching ' + str(result) + ' for ' + str(scenario_name) + '.')
    print('***Done fetching ' + str(result) + '.')

def process_reeds_data(topwdg, custom_sorts, custom_colors, result_dfs):
    '''
    Apply joins, mappings, ordering data to a selected result dataframe.
    Also categorize the columns of the dataframe and fill NA values.

    Args:
        topwdg (ordered dict): ReEDS widgets (meta widgets, scenarios widget, result widget)
        custom_sorts (dict): Keys are column names. Values are lists of values in the desired sort order.
        custom_colors (dict): Keys are column names and values are dicts that map column values to colors (hex strings)
        result_dfs (dict): Keys are ReEDS result names. Values are dataframes for that result (with 'scenario' as one of the columns)

    Returns:
        df (pandas dataframe): A dataframe of the ReEDS result, with filled NA values.
        cols (dict): Keys are categories of columns of df_source, and values are a list of columns of that category.
    '''
    print('***Apply joins, maps, ordering to ReEDS data...')
    df = result_dfs[topwdg['result'].value].copy()
    #apply joins
    for col in df.columns.values.tolist():
        if 'meta_join_'+col in topwdg and topwdg['meta_join_'+col].value != '':
            df_join = pd.read_csv(topwdg['meta_join_'+col].value.replace('"',''), dtype=object)
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
            df_map = pd.read_csv(topwdg['meta_map_'+col].value.replace('"',''), dtype=object)
            #filter out values that aren't in raw column
            df = df[df[col].isin(df_map['raw'].values.tolist())]
            #now map from raw to display
            map_dict = dict(zip(list(df_map['raw']), list(df_map['display'])))
            df[col] = df[col].map(map_dict)

    #apply custom styling
    for col in df.columns.values.tolist():
        custom_sorts.pop(col, None)
        custom_colors.pop(col, None)
        if 'meta_style_'+col in topwdg and topwdg['meta_style_'+col].value != '':
            df_style = pd.read_csv(topwdg['meta_style_'+col].value.replace('"',''), dtype=object)
            #filter out values that aren't in order column
            df = df[df[col].isin(df_style['order'].values.tolist())]
            #add to custom_sorts with new order
            custom_sorts[col] = df_style['order'].tolist()
            if 'color' in df_style:
                custom_colors[col] = dict(zip(df_style['order'],df_style['color']))
    cols = {}
    cols['all'] = df.columns.values.tolist()
    for c in cols['all']:
        if c in reeds.columns_meta:
            if reeds.columns_meta[c]['type'] is 'number':
                df[c] = pd.to_numeric(df[c], errors='coerce')
            elif reeds.columns_meta[c]['type'] is 'string':
                df[c] = df[c].astype(str)

    cols['discrete'] = [x for x in cols['all'] if df[x].dtype == object]
    cols['continuous'] = [x for x in cols['all'] if x not in cols['discrete']]
    cols['y-axis'] = [x for x in cols['continuous'] if x not in reeds.columns_meta or reeds.columns_meta[x]['y-allow']]
    cols['x-axis'] = [x for x in cols['all'] if x not in cols['y-axis']]
    cols['filterable'] = cols['discrete']+[x for x in cols['continuous'] if x in reeds.columns_meta and reeds.columns_meta[x]['filterable']]
    cols['seriesable'] = cols['discrete']+[x for x in cols['continuous'] if x in reeds.columns_meta and reeds.columns_meta[x]['seriesable']]
    df[cols['discrete']] = df[cols['discrete']].fillna('{BLANK}')
    df[cols['continuous']] = df[cols['continuous']].fillna(0)
    print('***Done with joins, maps, ordering.')
    return (df, cols)

def build_reeds_presets_wdg(preset_options):
    '''
    Build the presets widget

    Args:
        preset_options (list): List of strings for preset selections.

    Returns:
        wdg (ordered dict): Dictionary of bokeh.model.widgets, with 'presets' as the only key.
    '''
    wdg = collections.OrderedDict()
    wdg['presets'] = bmw.Select(title='Presets', value='None', options=['None'] + preset_options, css_classes=['wdgkey-presets'])
    wdg['presets'].on_change('value', update_reeds_presets)
    return wdg

def update_reeds_data_source(path, init_load, init_config):
    '''
    Respond to updates of data_source_wdg which are identified as ReEDS paths

    Args:
        path: The updated value of data_source_wdg
        init_load (boolean): True if this is the initial load of the page
        init_config (dict): Initial configuration supplied by the URL.

    Returns:
        Nothing: Core Globals are updated
    '''
    core.GL['variant_wdg'], GL_REEDS['scenarios'] = get_wdg_reeds(path, init_load, init_config, core.GL['wdg_defaults'], core.GL['custom_sorts'])
    core.GL['widgets'].update(core.GL['variant_wdg'])
    #if this is the initial load, we need to build the rest of the widgets if we've selected a result.
    if init_load and core.GL['variant_wdg']['result'].value is not 'None':
        get_reeds_data(core.GL['variant_wdg'], GL_REEDS['scenarios'], GL_REEDS['result_dfs'])
        core.GL['df_source'], core.GL['columns'] = process_reeds_data(core.GL['variant_wdg'], core.GL['custom_sorts'], core.GL['custom_colors'], GL_REEDS['result_dfs'])
        preset_options = []
        if 'presets' in reeds.results_meta[core.GL['variant_wdg']['result'].value]:
            preset_options = reeds.results_meta[core.GL['variant_wdg']['result'].value]['presets'].keys()
        core.GL['widgets'].update(build_reeds_presets_wdg(preset_options))
        core.GL['widgets'].update(core.build_widgets(core.GL['df_source'], core.GL['columns'], init_load, init_config, wdg_defaults=core.GL['wdg_defaults']))

def update_reeds_meta(attr, old, new):
    '''
    When ReEDS meta fields are updated, call update_reeds_wdg with the 'meta' flag
    '''
    update_reeds_wdg(type='meta')

def update_reeds_result(attr, old, new):
    '''
    When ReEDS Result field is updated, call update_reeds_wdg with the 'result' flag
    '''
    update_reeds_wdg(type='result')

def update_reeds_wdg(type):
    '''
    When ReEDS result field or meta field are updated, build core widgets accordingly
    
    Args:
        type (string): 'meta' or 'result'. Indicates the type of widget that was changed.
    '''
    core.GL['widgets'] = core.GL['data_source_wdg'].copy()
    core.GL['widgets'].update(core.GL['variant_wdg'])
    for key in core.GL['wdg_defaults'].keys():
        if key not in core.GL['variant_wdg'].keys() + ['data']:
            core.GL['wdg_defaults'].pop(key, None)
    if 'result' in core.GL['variant_wdg'] and core.GL['variant_wdg']['result'].value is not 'None':
        preset_options = []
        if type == 'result':
            get_reeds_data(core.GL['variant_wdg'], GL_REEDS['scenarios'], GL_REEDS['result_dfs'])
        if 'presets' in reeds.results_meta[core.GL['variant_wdg']['result'].value]:
            preset_options = reeds.results_meta[core.GL['variant_wdg']['result'].value]['presets'].keys()
        core.GL['df_source'], core.GL['columns'] = process_reeds_data(core.GL['variant_wdg'], core.GL['custom_sorts'], core.GL['custom_colors'], GL_REEDS['result_dfs'])
        core.GL['widgets'].update(build_reeds_presets_wdg(preset_options))
        core.GL['widgets'].update(core.build_widgets(core.GL['df_source'], core.GL['columns'], wdg_defaults=core.GL['wdg_defaults']))
    core.GL['controls'].children = list(core.GL['widgets'].values())
    core.GL['plots'].children = []

def update_reeds_presets(attr, old, new):
    '''
    When ReEDS preset is selected, clear all filter and main selectors, and set them
    to the state specified in the preset in reeds.py
    '''
    wdg = core.GL['widgets']
    preset = reeds.results_meta[wdg['result'].value]['presets'][wdg['presets'].value]
    core.preset_wdg(preset)