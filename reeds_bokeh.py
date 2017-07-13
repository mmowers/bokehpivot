from __future__ import division
import os
import pandas as pd
import collections
import bokeh.models.widgets as bmw
import gdx2py
import reeds
import core

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
        topwdg['result'] = bmw.Select(title='Result', value='None', options=['None']+list(reeds.results_meta.keys()), css_classes=['wdgkey-result'])
    #save defaults
    core.save_wdg_defaults(topwdg, wdg_defaults)
    #set initial config
    if init_load:
        core.initialize_wdg(topwdg, wdg_config)
    #Add update functions
    for key in topwdg:
        if key.startswith('meta_'):
            topwdg[key].on_change('value', core.update_reeds_meta)
    topwdg['result'].on_change('value', core.update_reeds_result)
    
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

def process_reeds_data(topwdg, custom_sorts, result_dfs):
    '''
    Apply joins, mappings, ordering data to a selected result dataframe.
    Also categorize the columns of the dataframe and fill NA values.

    Args:
        topwdg (ordered dict): ReEDS widgets (meta widgets, scenarios widget, result widget)
        custom_sorts (dict): Keys are column names. Values are lists of values in the desired sort order.
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
        if 'meta_style_'+col in topwdg and topwdg['meta_style_'+col].value != '':
            df_style = pd.read_csv(topwdg['meta_style_'+col].value.replace('"',''), dtype=object)
            #filter out values that aren't in order column
            df = df[df[col].isin(df_style['order'].values.tolist())]
            #add to custom_sorts with new order
            custom_sorts[col] = df_style['order'].tolist()
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