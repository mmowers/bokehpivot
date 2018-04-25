'''
Pivot chart maker core functionality and csv, gdx applications

'''
from __future__ import division
import os
import shutil
import re
import math
import json
import getpass
import numpy as np
import pandas as pd
import collections
import bokeh.io as bio
import bokeh.layouts as bl
import bokeh.models.widgets as bmw
import bokeh.models.sources as bms
import bokeh.models.tools as bmt
import bokeh.plotting as bp
import bokeh.palettes as bpa
import bokeh.resources as br
import bokeh.embed as be
import datetime
import six.moves.urllib.parse as urlp
import gdx2py
import subprocess as sp
import jinja2 as ji
import reeds_bokeh as rb

#Debug mode
debug = False

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
BAR_WIDTH = 0.8
LINE_WIDTH = 2
#colors taken from bpa.all_palettes['Category20'][20] and rearranged so that pairs are split
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5']*1000
MAP_PALETTE = 'Blues' #See https://bokeh.pydata.org/en/latest/docs/reference/palettes.html for options
C_NORM = "#31AADE"
CHARTTYPES = ['Dot', 'Line', 'Bar', 'Area', 'Map']
STACKEDTYPES = ['Bar', 'Area']
AGGREGATIONS = ['None', 'Sum', 'Ave', 'Weighted Ave', 'Weighted Ave Ratio']
ADV_BASES = ['Consecutive', 'Total']
MAP_FONT_SIZE = 10
MAP_NUM_BINS = 9
MAP_WIDTH = 500
MAP_OPACITY = 1
MAP_LINE_WIDTH = 0.1
RANGE_OPACITY_MULT = 0.3
RANGE_GLYPH_MAP = {'Line': 'Area', 'Dot': 'Bar'}

#List of widgets that use columns as their selectors
WDG_COL = ['x', 'y', 'x_group', 'series', 'explode', 'explode_group']

#List of widgets that don't use columns as selector and share general widget update function
WDG_NON_COL = ['chart_type', 'range', 'y_agg', 'y_weight', 'y_weight_denom', 'adv_op', 'adv_col_base', 'plot_title', 'plot_title_size',
    'plot_width', 'plot_height', 'opacity', 'sync_axes', 'x_min', 'x_max', 'x_scale', 'x_title',
    'x_title_size', 'x_major_label_size', 'x_major_label_orientation',
    'y_min', 'y_max', 'y_scale', 'y_title', 'y_title_size', 'y_major_label_size',
    'circle_size', 'bar_width', 'line_width', 'range_show_glyphs', 'net_levels', 'bokeh_tools', 'map_bin', 'map_num', 'map_min', 'map_max', 'map_manual',
    'map_width', 'map_font_size', 'map_line_width', 'map_opacity', 'map_palette', 'map_palette_2', 'map_palette_break']

#initialize globals dict for variables that are modified within update functions.
#custom_sorts: keys are column names. Values are lists of values in the desired sort order
#custom_colors (dict): Keys are column names and values are dicts that map column values to colors (hex strings)
GL = {'df_source':None, 'df_plots':None, 'columns':None, 'data_source_wdg':None, 'variant_wdg':{},
      'widgets':None, 'wdg_defaults': collections.OrderedDict(), 'controls': None, 'plots':None, 'custom_sorts': {},
      'custom_colors': {}}

#os globals
this_dir_path = os.path.dirname(os.path.realpath(__file__))
username = getpass.getuser()
user_out_path = this_dir_path + '/out/' + username
if not os.path.exists(user_out_path):
    os.makedirs(user_out_path)

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
    GL['wdg_defaults']['data'] = ''
    if wdg_arr is not None:
        wdg_config = json.loads(urlp.unquote(wdg_arr[0].decode('utf-8')))
        if 'data' in wdg_config:
            data_source = str(wdg_config['data'])

    #build widgets and plots
    GL['data_source_wdg'] = build_data_source_wdg(data_source)
    GL['controls'] = bl.widgetbox(list(GL['data_source_wdg'].values()), css_classes=['widgets_section'])
    GL['plots'] = bl.column([], css_classes=['plots_section'])
    layout = bl.row(GL['controls'], GL['plots'], css_classes=['full_layout'])

    if data_source != '':
        update_data_source(init_load=True, init_config=wdg_config)
        set_wdg_col_options()
        update_plots()

    bio.curdoc().add_root(layout)
    bio.curdoc().title = "Exploding Pivot Chart Maker"
    print('***Done Initializing')

def static_report(data_source, static_presets, report_path, report_format, html_num, output_dir, auto_open, variant_wdg_config=[]):
    '''
    Build static HTML and excel reports based on specified presets.
    Args:
        data_source (string): Path to data for which a report will be made
        static_presets (list of dicts): List of presets for which to make report. Each preset has these keys:
            'name': name of preset
            'config': a dict of widget configurations, where keys are keys of GL['widgets'] and values are values of those widgets. See preset_wdg()
        report_path (string): The path to the report file.
        report_format (string): 'html', 'excel', or 'both', specifying which reports to make
        html_num (string): 'multiple' if we are building separate html reports for each section, and 'one' for one html report with all sections.
        output_dir (string): the directory into which the resulting reports will be saved.
        auto_open (string): either "yes" to automatically open report files, or "no"
        variant_wdg_config (list of dicts): After data source is set, this allows us to set any other variant_wdg values.
    Returns:
        Nothing: HTML and Excel files are created
    '''
    #build initial widgets and plots globals
    GL['data_source_wdg'] = build_data_source_wdg('')
    GL['controls'] = bl.widgetbox(list(GL['data_source_wdg'].values()))
    GL['plots'] = bl.column([])
    #Update data source widget with input value
    GL['data_source_wdg']['data'].value = data_source
    #update any variant_wdg
    for vwc in variant_wdg_config:
        if vwc['type'] == 'active':
            GL['widgets'][vwc['name']].active = vwc['val']
    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = output_dir + '/'
    os.makedirs(output_dir)
    #copy report file to output_dir
    if report_path != '':
        shutil.copy2(report_path, output_dir)
    data_sources = data_source.split('|')
    if report_format in ['excel', 'both']:
        excel_report_path = output_dir + 'report.xlsx'
        excel_report = pd.ExcelWriter(excel_report_path)
        excel_meta = []
        excel_meta.append('Build date/time: ' + time)
        excel_meta.append('Data Source(s):')
        for ds in data_sources:
            excel_meta.append(ds)
        pd.Series(excel_meta).to_excel(excel_report, 'meta', index=False, header=False)
    if report_format in ['html', 'both']:
        with open(this_dir_path + '/templates/static/index.html', 'r') as template_file:
            template_string=template_file.read()
        template = ji.Template(template_string)
        resources = br.Resources()
        header = '<h3>Build date/time:</h3><p>' + time + '</p>'
        header += '<h3>Data Source(s):</h3><ul>'
        for ds in data_sources:
            header += '<li>' + ds + '</li>'
        header += '</ul>'
        header_row = bl.row(bmw.Div(text=header))
        if html_num == 'one':
            static_plots = []
            static_plots.append(header_row)
    #for each preset, set the widgets in preset_wdg(). Gather plots into separate sections of the html report,
    #and gather data into separate sheets of excel report
    sec_i = 1
    for static_preset in static_presets:
        name = static_preset['name']
        print('***Building report section: ' + name + '...')
        preset = static_preset['config']
        preset_wdg(preset)
        if report_format in ['html', 'both']:
            title = bmw.Div(text='<h2>' + str(sec_i) + '. ' + name + '</h2>')
            legend = bmw.Div(text=GL['widgets']['legend'].text)
            title_row = bl.row(title)
            content_row = bl.row(GL['plots'].children + [legend])
            if html_num == 'one':
                static_plots.append(title_row)
                static_plots.append(content_row)
            elif html_num == 'multiple':
                html = be.file_html([header_row, title_row, content_row], resources=resources, template=template)
                html_file_name = str(sec_i) + '_' + name
                html_file_name = re.sub(r'[\\/:"*?<>|]', '-', html_file_name) #replace disallowed file name characters with dash
                html_path = output_dir + html_file_name + '.html'
                with open(html_path, 'w') as f:
                    f.write(html)
                if auto_open == 'yes':
                    sp.Popen(os.path.abspath(html_path), shell=True)
        if report_format in ['excel', 'both']:
            excel_sheet_name = str(sec_i) + '_' + name
            excel_sheet_name = re.sub(r"[\\/*\[\]:?]", '-', excel_sheet_name) #replace disallowed sheet name characters with dash
            excel_sheet_name = excel_sheet_name[:31] #excel sheet names can only be 31 characters long
            GL['df_plots'].to_excel(excel_report, excel_sheet_name, index=False)
        sec_i += 1
    if report_format in ['excel', 'both']:
        excel_report.save()
        if auto_open == 'yes':
            sp.Popen(os.path.abspath(excel_report_path), shell=True)
    if report_format in ['html', 'both'] and html_num == 'one':
        html = be.file_html(static_plots, resources=resources, template=template)
        html_path = output_dir + 'report.html'
        with open(html_path, 'w') as f:
            f.write(html)
        if auto_open == 'yes':
            sp.Popen(os.path.abspath(html_path), shell=True)
    print('***Done building report')

def preset_wdg(preset):
    '''
    Reset widgets and then set them to that specified in input preset
    Args:
        preset (dict): keys are widget names, and values are the desired widget values. Filters are entered as a list of labels under the 'filter' key.
    Returns:
        Nothing: widget values are set.
    '''
    #First set all wdg_variant values, if they exist, in order that they appear in wdg_variant, an ordered dict.
    variant_presets = [key for key in GL['variant_wdg'].keys() if key in preset]
    for key in variant_presets:
        if isinstance(GL['widgets'][key], bmw.groups.Group):
            GL['widgets'][key].active = [GL['widgets'][key].labels.index(i) for i in preset[key]]
        elif isinstance(GL['widgets'][key], bmw.inputs.InputWidget):
            GL['widgets'][key].value = preset[key]
    #these variables are set after the variant_wdg presets because otherwise they diverge from the globals
    wdg = GL['widgets']
    wdg_variant = GL['variant_wdg']
    wdg_defaults = GL['wdg_defaults']
    #Now set x to none to prevent chart rerender
    wdg['x'].value = 'None'
    #gather widgets to reset
    wdg_resets = [i for i in wdg_defaults if i not in wdg_variant.keys()+['x', 'data', 'render_plots', 'auto_update']]
    #reset widgets if they are not default
    for key in wdg_resets:
        if isinstance(wdg[key], bmw.groups.Group) and wdg[key].active != wdg_defaults[key]:
            wdg[key].active = wdg_defaults[key]
        elif isinstance(wdg[key], bmw.inputs.InputWidget) and wdg[key].value != wdg_defaults[key]:
            wdg[key].value = wdg_defaults[key]
    #set all presets except x and filter, in order that they appear in wdg, an ordered dict.
    #Filters are handled separately, after that. x will be set at end, triggering render of chart.
    common_presets = [key for key in wdg.keys() if key in preset and key not in wdg_variant.keys()+['x', 'filter']]
    for key in common_presets:
        if isinstance(wdg[key], bmw.groups.Group):
            wdg[key].active = [wdg[key].labels.index(i) for i in preset[key]]
        elif isinstance(wdg[key], bmw.inputs.InputWidget):
            wdg[key].value = preset[key]
    #filters are handled separately. We must deal with the active arrays of each filter
    if 'filter' in preset:
        for fil in preset['filter']:
            preset_filter = preset['filter'][fil]
            #find index of associated filter:
            for j, col in enumerate(GL['columns']['filterable']):
                if col == fil:
                    #get filter widget associated with found index
                    wdg_fil = wdg['filter_'+str(j)]
                    #build the new_active list, starting with zeros
                    #for each label given in the preset, set corresponding active to 1
                    if isinstance(preset_filter, str):
                        if preset_filter == 'last':
                            new_active = [len(wdg_fil.labels) - 1]
                    elif isinstance(preset_filter, dict):
                        new_active = list(range(len(wdg_fil.labels)))
                        if 'start' in preset_filter:
                            start = wdg_fil.labels.index(str(preset_filter['start']))
                            if 'end' in preset_filter:
                                end = wdg_fil.labels.index(str(preset_filter['end']))
                            else:
                                end = len(wdg_fil.labels) - 1
                            new_active = list(range(start,end+1))
                        if 'exclude' in preset_filter:
                            new_active = [n for n in new_active if wdg_fil.labels[n] not in preset_filter['exclude']]
                    else: #we are using a list of labels
                        new_active = []
                        for lab in preset_filter:
                            if str(lab) in wdg_fil.labels:
                                index = wdg_fil.labels.index(str(lab))
                                new_active.append(index)
                    wdg_fil.active = new_active
                    break
    #finally, set x, which will trigger the data and chart updates.
    wdg['x'].value = preset['x']

def build_data_source_wdg(data_source):
    '''
    Return the initial data source widget, prefilled with an input data_source
    Args:
        data_source (string): Path to data source
    Returns:
        wdg (ordered dict): ordered dictionary of bokeh.models.widgets (in this case only one) for data source.
    '''
    wdg = collections.OrderedDict()
    wdg['readme'] = bmw.Div(text='<a href="https://github.nrel.gov/ReEDS/bokehpivot" target="_blank">README</a>')
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

def build_widgets(df_source, cols, init_load=False, init_config={}, wdg_defaults={}):
    '''
    Use a dataframe and its columns to set widget options. Widget values may
    be set by URL parameters via init_config.

    Args:
        df_source (pandas dataframe): Dataframe of the csv source.
        cols (dict): Keys are categories of columns of df_source, and values are a list of columns of that category.
        init_load (boolean, optional): If this is the initial page load, then this will be True, else False.
        init_config (dict): Initial widget configuration passed via URL.
        wdg_defaults (dict): Keys are widget names and values are the default values of the widgets.

    Returns:
        wdg (ordered dict): Dictionary of bokeh.model.widgets.
    '''
    #Add widgets
    print('***Build main widgets...')
    wdg = collections.OrderedDict()
    wdg['chart_type_dropdown'] = bmw.Div(text='Chart', css_classes=['chart-dropdown'])
    wdg['chart_type'] = bmw.Select(title='Chart Type', value='Dot', options=CHARTTYPES, css_classes=['wdgkey-chart_type', 'chart-drop'])
    wdg['range'] = bmw.Select(title='Add Ranges (Line and Dot only)', value='No', options=['No', 'Within Series', 'Between Series'], css_classes=['wdgkey-range', 'chart-drop'])
    wdg['x_dropdown'] = bmw.Div(text='X-Axis (required)', css_classes=['x-dropdown'])
    wdg['x'] = bmw.Select(title='X-Axis (required)', value='None', options=['None'] + cols['x-axis'], css_classes=['wdgkey-x', 'x-drop'])
    wdg['x_group'] = bmw.Select(title='Group X-Axis By', value='None', options=['None'] + cols['seriesable'], css_classes=['wdgkey-x_group', 'x-drop'])
    wdg['y_dropdown'] = bmw.Div(text='Y-Axis (required)', css_classes=['y-dropdown'])
    wdg['y'] = bmw.Select(title='Y-Axis (required)', value='None', options=['None'] + cols['y-axis'], css_classes=['wdgkey-y', 'y-drop'])
    wdg['y_agg'] = bmw.Select(title='Y-Axis Aggregation', value='Sum', options=AGGREGATIONS, css_classes=['wdgkey-y_agg', 'y-drop'])
    wdg['y_weight'] = bmw.Select(title='Weighting Factor', value='None', options=['None'] + cols['y-axis'], css_classes=['wdgkey-y_weight', 'y-drop'])
    wdg['y_weight_denom'] = bmw.Select(title='Denominator Weighting Factor', value='None', options=['None'] + cols['y-axis'], css_classes=['wdgkey-y_weight_denom', 'y-drop'])
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
    wdg['plot_width'] = bmw.TextInput(title='Plot Width (px)', value=str(PLOT_WIDTH), css_classes=['wdgkey-plot_width', 'adjust-drop'])
    wdg['plot_height'] = bmw.TextInput(title='Plot Height (px)', value=str(PLOT_HEIGHT), css_classes=['wdgkey-plot_height', 'adjust-drop'])
    wdg['plot_title'] = bmw.TextInput(title='Plot Title', value='', css_classes=['wdgkey-plot_title', 'adjust-drop'])
    wdg['plot_title_size'] = bmw.TextInput(title='Plot Title Font Size', value=str(PLOT_FONT_SIZE), css_classes=['wdgkey-plot_title_size', 'adjust-drop'])
    wdg['opacity'] = bmw.TextInput(title='Opacity (0-1)', value=str(OPACITY), css_classes=['wdgkey-opacity', 'adjust-drop'])
    wdg['sync_axes'] = bmw.Select(title='Sync Axes', value='Yes', options=['Yes', 'No'], css_classes=['adjust-drop'])
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
    wdg['bar_width_desc'] = bmw.Div(text='<strong>Flags</strong> <em>w</em>: use csv file for widths, <em>c</em>: sort and convert x axis to quantitative based on csv file, <em>s</em>: sort bars according to height (include numeric width)', css_classes=['adjust-drop', 'description'])
    wdg['line_width'] = bmw.TextInput(title='Line Width (Line Only)', value=str(LINE_WIDTH), css_classes=['wdgkey-line_width', 'adjust-drop'])
    wdg['range_show_glyphs'] = bmw.Select(title='Show Line/Dot (Range Only)', value='Yes', options=['Yes','No'], css_classes=['wdgkey-range_show_glyphs', 'adjust-drop'])
    wdg['net_levels'] = bmw.Select(title='Add Net Levels to Stacked', value='Yes', options=['Yes','No'], css_classes=['wdgkey-net_levels', 'adjust-drop'])
    wdg['bokeh_tools'] = bmw.Select(title='Show Bokeh Tools', value='Yes', options=['Yes','No'], css_classes=['wdgkey-bokeh_tools', 'adjust-drop'])
    wdg['map_adjustments'] = bmw.Div(text='Map Adjustments', css_classes=['map-dropdown'])
    wdg['map_bin'] = bmw.Select(title='Bin Type', value='Auto Equal Num', options=['Auto Equal Num', 'Auto Equal Width', 'Manual'], css_classes=['wdgkey-map_bin', 'map-drop'])
    wdg['map_num'] = bmw.TextInput(title='# of bins (Auto Only)', value=str(MAP_NUM_BINS), css_classes=['wdgkey-map_num', 'map-drop'])
    wdg['map_palette'] = bmw.TextInput(title='Map Palette', value=MAP_PALETTE, css_classes=['wdgkey-map_palette', 'map-drop'])
    wdg['map_palette_desc'] = bmw.Div(text='See <a href="https://bokeh.pydata.org/en/latest/docs/reference/palettes.html" target="_blank">palette options</a> or all_red, all_blue, all_green, all_gray. Palette must accommodate # of bins', css_classes=['map-drop', 'description'])
    wdg['map_palette_2'] = bmw.TextInput(title='Map Palette 2 (Optional)', value='', css_classes=['wdgkey-map_palette_2', 'map-drop'])
    wdg['map_palette_2_desc'] = bmw.Div(text='Bins will be split between palettes.', css_classes=['map-drop', 'description'])
    wdg['map_palette_break'] = bmw.TextInput(title='Dual Palette Breakpoint (Optional)', value='', css_classes=['wdgkey-map_palette_break', 'map-drop'])
    wdg['map_palette_break_desc'] = bmw.Div(text='The bin that contains this breakpoint will divide the palettes.', css_classes=['map-drop', 'description'])
    wdg['map_min'] = bmw.TextInput(title='Minimum (Equal Width Only)', value='', css_classes=['wdgkey-map_min', 'map-drop'])
    wdg['map_max'] = bmw.TextInput(title='Maximum (Equal Width Only)', value='', css_classes=['wdgkey-map_max', 'map-drop'])
    wdg['map_manual'] = bmw.TextInput(title='Manual Breakpoints (Manual Only)', value='', css_classes=['wdgkey-map_manual', 'map-drop'])
    wdg['map_manual_desc'] = bmw.Div(text='Comma separated list of values (e.g. -10,0,0.5,6), with one fewer value than # of bins', css_classes=['map-drop', 'description'])
    wdg['map_width'] = bmw.TextInput(title='Map Width (px)', value=str(MAP_WIDTH), css_classes=['wdgkey-map_width', 'map-drop'])
    wdg['map_font_size'] = bmw.TextInput(title='Title Font Size', value=str(MAP_FONT_SIZE), css_classes=['wdgkey-map_font_size', 'map-drop'])
    wdg['map_line_width'] = bmw.TextInput(title='Line Width', value=str(MAP_LINE_WIDTH), css_classes=['wdgkey-map_line_width', 'map-drop'])
    wdg['map_opacity'] = bmw.TextInput(title='Opacity (0-1)', value=str(MAP_OPACITY), css_classes=['wdgkey-map_opacity', 'map-drop'])
    wdg['auto_update_dropdown'] = bmw.Div(text='Auto/Manual Update', css_classes=['update-dropdown'])
    wdg['auto_update'] = bmw.Select(title='Auto Update (except filters)', value='Enable', options=['Enable', 'Disable'], css_classes=['update-drop'])
    wdg['update'] = bmw.Button(label='Manual Update', button_type='success', css_classes=['update-drop'])
    wdg['render_plots'] = bmw.Select(title='Render Plots', value='Yes', options=['Yes', 'No'], css_classes=['update-drop'])
    wdg['download_dropdown'] = bmw.Div(text='Download/Export', css_classes=['download-dropdown'])
    wdg['download_all'] = bmw.Button(label='All Files of View', button_type='success', css_classes=['download-drop'])
    wdg['download_csv'] = bmw.Button(label='CSV of View', button_type='success', css_classes=['download-drop'])
    wdg['download_html'] = bmw.Button(label='HTML of View', button_type='success', css_classes=['download-drop'])
    wdg['download_url'] = bmw.Button(label='URL of View', button_type='success', css_classes=['download-drop'])
    wdg['download_report'] = bmw.Button(label='Python Report of View', button_type='success', css_classes=['download-drop'])
    wdg['download_reeds_preset'] = bmw.Button(label='ReEDS Preset of View', button_type='success', css_classes=['download-drop'])
    wdg['download_source'] = bmw.Button(label='CSV of Full Data Source', button_type='success', css_classes=['download-drop'])
    wdg['legend_dropdown'] = bmw.Div(text='Legend', css_classes=['legend-dropdown'])
    wdg['legend'] = bmw.Div(text='', css_classes=['legend-drop'])
    wdg['display_config'] = bmw.Div(text='', css_classes=['display-config'])

    #save defaults
    save_wdg_defaults(wdg, wdg_defaults)
    #use init_config (from 'widgets' parameter in URL query string) to configure widgets.
    if init_load:
        initialize_wdg(wdg, init_config)

    #Add update functions for widgets
    wdg['filters_update'].on_click(update_plots)
    wdg['update'].on_click(update_plots)
    wdg['download_all'].on_click(download_all)
    wdg['download_csv'].on_click(download_csv)
    wdg['download_html'].on_click(download_html)
    wdg['download_url'].on_click(download_url)
    wdg['download_report'].on_click(download_report)
    wdg['download_reeds_preset'].on_click(download_reeds_preset)
    wdg['download_source'].on_click(download_source)
    wdg['adv_col'].on_change('value', update_adv_col)
    for name in WDG_COL:
        wdg[name].on_change('value', update_wdg_col)
    for name in WDG_NON_COL:
        wdg[name].on_change('value', update_wdg)
    print('***Done with main widgets.')
    return wdg

def initialize_wdg(wdg, init_config):
    '''
    Set values of wdg based on init_config

    Args:
        wdg (ordered dict): Dictionary of bokeh.model.widgets.
        init_config (dict): Initial widget configuration passed via URL.

    Returns:
        Nothing: wdg is modified
    '''
    for key in init_config:
        if key in wdg:
            if hasattr(wdg[key], 'value'):
                wdg[key].value = str(init_config[key])
            elif hasattr(wdg[key], 'active'):
                wdg[key].active = init_config[key]

def save_wdg_defaults(wdg, wdg_defaults):
    '''
    Save wdg_defaults based on wdg

    Args:
        wdg (ordered dict): Dictionary of bokeh.model.widgets.
        wdg_defaults (dict): Keys are widget names and values are the default values of the widgets.

    Returns:
        Nothing: wdg_defaults is set for applicable keys in wdg
    '''
    for key in wdg:
        if isinstance(wdg[key], bmw.groups.Group):
            wdg_defaults[key] = wdg[key].active
        elif isinstance(wdg[key], bmw.inputs.InputWidget):
            wdg_defaults[key] = wdg[key].value

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
    startTime = datetime.datetime.now()
    df_plots = df_source.copy()

    #Apply filters
    for j, col in enumerate(cols['filterable']):
        active = [wdg['filter_'+str(j)].labels[i] for i in wdg['filter_'+str(j)].active]
        if col in cols['continuous']:
            active = [float(i) for i in active]
        df_plots = df_plots[df_plots[col].isin(active)]

    if df_plots.empty:
        return df_plots

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
        kwargs = {}
        if wdg['y_agg'].value == 'Weighted Ave' and wdg['y_weight'].value in cols['continuous']:
            kwargs['y_weight'] = wdg['y_weight'].value
        elif wdg['y_agg'].value == 'Weighted Ave Ratio' and wdg['y_weight'].value in cols['continuous'] and wdg['y_weight_denom'].value in cols['continuous']:
            kwargs['y_weight_numer'] = wdg['y_weight'].value
            kwargs['y_weight_denom'] = wdg['y_weight_denom'].value
        df_plots = df_grouped.apply(apply_aggregation, wdg['y_agg'].value, wdg['y'].value, wdg['range'].value, kwargs).reset_index()
        #The index of each group's dataframe is added as another column it seems. So we need to remove it:
        df_plots.drop(df_plots.columns[len(groupby_cols)], axis=1, inplace=True)

    #Check for range chart
    range_cols = []
    if wdg['range'].value == 'Within Series':
        range_cols = ['range_min', 'range_max']

    #Do Advanced Operations
    op = wdg['adv_op'].value
    col = wdg['adv_col'].value
    col_base = wdg['adv_col_base'].value
    y_val = wdg['y'].value
    y_agg = wdg['y_agg'].value
    if op != 'None' and col != 'None' and col in df_plots and col_base != 'None' and y_agg != 'None' and y_val in cols['continuous'] and wdg['range'].value == 'No':
        if col in cols['continuous'] and col_base not in ADV_BASES:
            col_base = float(col_base)
        #groupby all columns that are not the operating column and y axis column so we can do operations on y-axis across the operating column
        groupcols = [i for i in df_plots.columns.values.tolist() if i not in [col, y_val]]
        if groupcols != []:
            df_grouped = df_plots.groupby(groupcols, sort=False)
        else:
            #if we don't have other columns to group, make one, to prevent error
            df_plots['tempgroup'] = 1
            df_grouped = df_plots.groupby('tempgroup', sort=False)
        #Now do operations with the groups:
        df_plots = df_grouped.apply(op_with_base, op, col, col_base, y_val).reset_index(drop=True)
        df_plots = df_plots.replace([np.inf, -np.inf], np.nan)
        #Finally, clean up df_plots, dropping unnecessary columns, rows with the base value, and any rows with NAs for y_vals
        if 'tempgroup' in df_plots:
            df_plots.drop(['tempgroup'], axis='columns', inplace=True)
        df_plots = df_plots[~df_plots[col].isin([col_base])]
        df_plots = df_plots[pd.notnull(df_plots[y_val])]

    #For bar charts with convert flag 'c' or sort flag 's', we will sort by cumulative y value.
    bar_height_sort = False
    if wdg['chart_type'].value == 'Bar' and (wdg['bar_width'].value == 'c'  or 's' in wdg['bar_width'].value):
        bar_height_sort = True
        #adjust groupby_cols from Aggregation section above, and remove series from group if it is there
        bar_group_cols = [c for c in groupby_cols if c != wdg['series'].value]
        #group and sum across series to get the cumulative y for each x
        df_bar_group = df_plots.groupby(bar_group_cols, sort=False)
        df_bar = df_bar_group[wdg['y'].value].sum().reset_index()
        df_bar.rename(columns={wdg['y'].value: 'y_bar_cumulative'}, inplace=True)
        #multiply by -1 so that we sort from large to small instead of small to large
        df_bar['y_bar_cumulative'] = df_bar['y_bar_cumulative']*-1
        df_plots = df_plots.merge(df_bar, how='left', on=bar_group_cols, sort=False)

    #Sort Dataframe
    sortby_cols = [wdg['x'].value]
    if wdg['x_group'].value != 'None': sortby_cols = [wdg['x_group'].value] + sortby_cols
    if bar_height_sort: sortby_cols = ['y_bar_cumulative'] + sortby_cols
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
    if bar_height_sort:
        df_plots = df_plots.drop('y_bar_cumulative', 1)
        sortby_cols.remove('y_bar_cumulative')

    #Rearrange column order for csv download
    sorted_cols = sortby_cols + [wdg['y'].value] + range_cols
    unsorted_columns = [col for col in df_plots.columns if col not in sorted_cols]
    df_plots = df_plots[unsorted_columns + sorted_cols]
    print('***Done Filtering, Scaling, Aggregating, Adv Operations, Sorting: '+ str(datetime.datetime.now() - startTime))
    if wdg['render_plots'].value == 'No':
        print('***Ready for download!')
    return df_plots

def create_figures(df_plots, wdg, cols, custom_colors):
    '''
    Create figures based on the data in a dataframe and widget configuration, and return figures in a list.
    The explode widget determines if there will be multiple figures.

    Args:
        df_plots (pandas dataframe): Dataframe of csv source after being filtered, scaled, aggregated, and sorted.
        wdg (ordered dict): Dictionary of bokeh model widgets.
        cols (dict): Keys are categories of columns of df_source, and values are a list of columns of that category.
        custom_colors (dict): Keys are column names and values are dicts that map column values to colors (hex strings)

    Returns:
        plot_list (list): List of bokeh.model.figures.
    '''
    print('***Building Figures...')
    plot_list = []
    df_plots_cp = df_plots.copy()
    if wdg['explode'].value == 'None':
        plot_list.append(create_figure(df_plots_cp, df_plots, wdg, cols, custom_colors))
    else:
        if wdg['explode_group'].value == 'None':
            for explode_val in df_plots_cp[wdg['explode'].value].unique().tolist():
                df_exploded = df_plots_cp[df_plots_cp[wdg['explode'].value].isin([explode_val])]
                plot_list.append(create_figure(df_exploded, df_plots, wdg, cols, custom_colors, explode_val))
        else:
            for explode_group in df_plots_cp[wdg['explode_group'].value].unique().tolist():
                df_exploded_group = df_plots_cp[df_plots_cp[wdg['explode_group'].value].isin([explode_group])]
                for explode_val in df_exploded_group[wdg['explode'].value].unique().tolist():
                    df_exploded = df_exploded_group[df_exploded_group[wdg['explode'].value].isin([explode_val])]
                    plot_list.append(create_figure(df_exploded, df_plots, wdg, cols, custom_colors, explode_val, explode_group))
    set_axis_bounds(df_plots, plot_list, wdg, cols)
    print('***Done Building Figures.')
    return plot_list

def set_axis_bounds(df, plots, wdg, cols):
    '''
    Set minimums and maximums for x and y axes.

    Args:
        df (pandas dataframe): Dataframe of all data currently being viewed.
        plots (list): List of bokeh.model.figures.
        wdg (ordered dict): Dictionary of bokeh model widgets.
        cols (dict): Keys are categories of columns of df_source, and values are a list of columns of that category.

    Returns:
        Nothing. Axes of plots are modified.
    '''
    if wdg['x'].value in cols['continuous'] and wdg['x_group'].value == 'None':
        if wdg['x_min'].value != '':
            for p in plots:
                p.x_range.start = float(wdg['x_min'].value)
        elif wdg['sync_axes'].value == 'Yes':
            min_x = df[wdg['x'].value].min()
            if wdg['chart_type'].value == 'Bar':
                min_x = min_x - float(wdg['bar_width'].value)/2
            for p in plots:
                p.x_range.start = min_x
        if wdg['x_max'].value != '':
            for p in plots:
                p.x_range.end = float(wdg['x_max'].value)
        elif wdg['sync_axes'].value == 'Yes':
            max_x = df[wdg['x'].value].max()
            if wdg['chart_type'].value == 'Bar':
                max_x = max_x + float(wdg['bar_width'].value)/2
            for p in plots:
                p.x_range.end = max_x
    elif wdg['chart_type'].value == 'Bar' and wdg['bar_width'].value == 'c':
        if wdg['x_min'].value != '':
            for p in plots:
                p.x_range.start = float(wdg['x_min'].value)
        if wdg['x_max'].value != '':
            for p in plots:
                p.x_range.end = float(wdg['x_max'].value)
    if wdg['y'].value in cols['continuous']:
        #find grouped cols for stacked manipulations
        col_names = df.columns.values.tolist()
        groupby_cols = [i for i in col_names if i not in [wdg['series'].value, wdg['y'].value]]
        if wdg['y_min'].value != '':
            for p in plots:
                p.y_range.start = float(wdg['y_min'].value)
        elif wdg['sync_axes'].value == 'Yes':
            if wdg['chart_type'].value in STACKEDTYPES:
                #sum negative values across series
                df_neg = df[df[wdg['y'].value] < 0]
                df_neg_sum = df_neg.groupby(groupby_cols, sort=False)[wdg['y'].value].sum().reset_index()
                min_y = df_neg_sum[wdg['y'].value].min()
            else:
                if wdg['range'].value == 'Within Series':
                    if wdg['range_show_glyphs'].value == 'Yes':
                        min_y = min(df['range_min'].min(), df[wdg['y'].value].min())
                    else:
                        min_y = df['range_min'].min()
                else:
                    min_y = df[wdg['y'].value].min()
            min_y = min(0, min_y)
            for p in plots:
                p.y_range.start = min_y
        if wdg['y_max'].value != '':
            for p in plots:
                p.y_range.end = float(wdg['y_max'].value)
        elif wdg['sync_axes'].value == 'Yes':
            if wdg['chart_type'].value in STACKEDTYPES:
                #sum postive values across series
                df_pos = df[df[wdg['y'].value] > 0]
                df_pos_sum = df_pos.groupby(groupby_cols, sort=False)[wdg['y'].value].sum().reset_index()
                max_y = df_pos_sum[wdg['y'].value].max()
            else:
                if wdg['range'].value == 'Within Series':
                    if wdg['range_show_glyphs'].value == 'Yes':
                        max_y = max(df['range_max'].max(), df[wdg['y'].value].max())
                    else:
                        max_y = df['range_max'].max()
                else:
                    max_y = df[wdg['y'].value].max()
            max_y = max(0, max_y)
            for p in plots:
                p.y_range.end = max_y


def create_figure(df_exploded, df_plots, wdg, cols, custom_colors, explode_val=None, explode_group=None):
    '''
    Create and return a figure based on the data in a dataframe and widget configuration.

    Args:
        df_exploded (pandas dataframe): Dataframe of just the data that will be plotted in this figure.
        df_plots (pandas dataframe): Dataframe of all plots data, used only for maintaining consistent series colors.
        wdg (ordered dict): Dictionary of bokeh model widgets.
        cols (dict): Keys are categories of columns of df_source, and values are a list of columns of that category.
        custom_colors (dict): Keys are column names and values are dicts that map column values to colors (hex strings)
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
    chart_type = wdg['chart_type'].value
    #Set x and y ranges. When x is grouped, there is added complication of separating the groups
    xs = df_exploded[x_col].values.tolist()
    ys = df_exploded[wdg['y'].value].values.tolist()
    if not (chart_type == 'Bar' and wdg['bar_width'].value == 'c'):
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
            kw['x_range'] = []
            for x in xs:
                if x not in kw['x_range']:
                    kw['x_range'].append(x)
        if wdg['y'].value in cols['discrete']:
            kw['y_range'] = []
            for y in ys:
                if y not in kw['y_range']:
                    kw['y_range'].append(y)

    #Set figure title
    kw['title'] = wdg['plot_title'].value
    seperator = '' if kw['title'] == '' else ', '
    if explode_val is not None:
        if explode_group is not None:
            kw['title'] = kw['title'] + seperator + str(explode_group)
        seperator = '' if kw['title'] == '' else ', '
        kw['title'] = kw['title'] + seperator + str(explode_val)

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
    if wdg['bokeh_tools'].value == 'No':
        p.toolbar.logo = None
        p.toolbar_location = None

    #Add glyphs to figure
    c = C_NORM
    if wdg['series'].value == 'None':
        if wdg['range'].value == 'Within Series':
            y_mins = df_exploded['range_min'].values.tolist()
            y_maxs = df_exploded['range_max'].values.tolist()
            add_glyph(RANGE_GLYPH_MAP[chart_type], wdg, p, xs, y_maxs, c, y_bases=y_mins, opacity_mult=RANGE_OPACITY_MULT)
        if wdg['range_show_glyphs'].value == 'Yes':
            add_glyph(chart_type, wdg, p, xs, ys, c)
    else:
        full_series = df_plots[wdg['series'].value].unique().tolist() #for colors only
        xs_full = df_exploded[x_col].unique().tolist()
        if chart_type in STACKEDTYPES: #We are stacking the series
            y_bases_pos = [0]*len(xs_full)
            y_bases_neg = [0]*len(xs_full)
        elif wdg['range'].value == 'Between Series':
            y_mins = []
            y_maxs = []
            for x_unique in xs_full:
                y_group = [ys[i] for i, x in enumerate(xs) if x == x_unique]
                y_mins.append(min(y_group))
                y_maxs.append(max(y_group))
            add_glyph(RANGE_GLYPH_MAP[chart_type], wdg, p, xs_full, y_maxs, c, y_bases=y_mins, opacity_mult=RANGE_OPACITY_MULT)
        for i, ser in enumerate(df_exploded[wdg['series'].value].unique().tolist()):
            if custom_colors and wdg['series'].value in custom_colors and ser in custom_colors[wdg['series'].value]:
                c = custom_colors[wdg['series'].value][ser]
            else:
                c = COLORS[full_series.index(ser)]
            df_series = df_exploded[df_exploded[wdg['series'].value].isin([ser])]
            xs_ser = df_series[x_col].values.tolist()
            ys_ser = df_series[wdg['y'].value].values.tolist()
            if chart_type not in STACKEDTYPES: #The series will not be stacked
                if wdg['range'].value == 'Within Series':
                    y_mins_ser = df_series['range_min'].values.tolist()
                    y_maxs_ser = df_series['range_max'].values.tolist()
                    add_glyph(RANGE_GLYPH_MAP[chart_type], wdg, p, xs_ser, y_maxs_ser, c, y_bases=y_mins_ser, series=ser, opacity_mult=RANGE_OPACITY_MULT)
                if wdg['range_show_glyphs'].value == 'Yes':
                    add_glyph(chart_type, wdg, p, xs_ser, ys_ser, c, series=ser)
            else: #We are stacking the series
                ys_pos = [ys_ser[xs_ser.index(x)] if x in xs_ser and ys_ser[xs_ser.index(x)] > 0 else 0 for i, x in enumerate(xs_full)]
                ys_neg = [ys_ser[xs_ser.index(x)] if x in xs_ser and ys_ser[xs_ser.index(x)] < 0 else 0 for i, x in enumerate(xs_full)]
                ys_stacked_pos = [ys_pos[i] + y_bases_pos[i] for i in range(len(xs_full))]
                ys_stacked_neg = [ys_neg[i] + y_bases_neg[i] for i in range(len(xs_full))]
                add_glyph(chart_type, wdg, p, xs_full, ys_stacked_pos, c, y_bases=y_bases_pos, series=ser)
                add_glyph(chart_type, wdg, p, xs_full, ys_stacked_neg, c, y_bases=y_bases_neg, series=ser)
                y_bases_pos = ys_stacked_pos
                y_bases_neg = ys_stacked_neg
        if wdg['net_levels'].value == 'Yes' and chart_type in STACKEDTYPES:
            ys_net = [ys_stacked_pos[i] + ys_stacked_neg[i] for i,x in enumerate(xs_full)]
            add_glyph('Dot', wdg, p, xs_full, ys_net, 'black', series='Net Level')
    return p

def add_glyph(glyph_type, wdg, p, xs, ys, c, y_bases=None, series=None, opacity_mult=1):
    '''
    Add a glyph to a Bokeh figure, depending on the chosen chart type.

    Args:
        glyph_type (str): Type of glyph (e.g. 'Dot', 'Line', 'Bar', 'Area')
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
    alpha = float(wdg['opacity'].value)*opacity_mult
    y_unstacked = list(ys) if y_bases is None else [ys[i] - y_bases[i] for i in range(len(ys))]
    ser = ['None']*len(xs) if series is None else [series]*len(xs)
    if glyph_type == 'Dot':
        source = bms.ColumnDataSource({'x': xs, 'y': ys, 'x_legend': xs, 'y_legend': y_unstacked, 'ser_legend': ser})
        p.circle('x', 'y', source=source, color=c, size=int(wdg['circle_size'].value), fill_alpha=alpha, line_color=None, line_width=None)
    elif glyph_type == 'Line':
        source = bms.ColumnDataSource({'x': xs, 'y': ys, 'x_legend': xs, 'y_legend': y_unstacked, 'ser_legend': ser})
        p.line('x', 'y', source=source, color=c, alpha=alpha, line_width=float(wdg['line_width'].value))
    elif glyph_type == 'Bar' and y_unstacked != [0]*len(y_unstacked):
        if y_bases is None: y_bases = [0]*len(ys)
        centers = [(ys[i] + y_bases[i])/2 for i in range(len(ys))]
        heights = [abs(ys[i] - y_bases[i]) for i in range(len(ys))]
        xs_cp = list(xs) #we don't want to modify xs that are passed into function
        x_legend = list(xs)
        if wdg['bar_width'].value == 'w': #this means we are looking for the mapping in the _bar_width file
            df_bar_width = pd.read_csv(this_dir_path + '/in/' + wdg['x'].value + '_bar_width.csv', index_col='m')
            max_width = df_bar_width['width'].max()
            widths = [df_bar_width.loc[x, 'width']/max_width for x in xs]
            x_legend = [x_legend[i] + ' (width = ' + str(df_bar_width.loc[x, 'width']) + ')' for i, x in enumerate(xs)]
        elif wdg['bar_width'].value == 'c': #this means we are converting x axis to continuous and have no gaps between bars
            widths = []
            df_bar_width = pd.read_csv(this_dir_path + '/in/' + wdg['x'].value + '_bar_width.csv', index_col='m')
            xs_cum = 0
            for i, xo in enumerate(xs):
                width = df_bar_width.loc[xo, 'width']
                widths.append(width)
                xs_cp[i] = width/2 + xs_cum
                xs_cum = xs_cum + width
                x_legend[i] = x_legend[i] + ' (width = ' + str(width) + ', ' + str(xs_cum) + ' cumulative)'
        else:
            #strip off letters, for example, the 's' flag.
            digits = re.sub("[a-z]", "", wdg['bar_width'].value)
            widths = [float(digits)]*len(xs)
        #bars have issues when height is 0, so remove elements whose height is 0 
        heights_orig = list(heights) #we make a copy so we aren't modifying the list we are iterating on.
        for i, h in reversed(list(enumerate(heights_orig))):
            #Ok this is getting absurd, but rects with near-zero heights also break the glyphs.
            #See https://github.com/bokeh/bokeh/issues/6583.
            if abs(h) <= 1e-13:
                del xs_cp[i], centers[i], heights[i], y_unstacked[i], ser[i], widths[i], x_legend[i]
        source = bms.ColumnDataSource({'x': xs_cp, 'y': centers, 'x_legend': x_legend, 'y_legend': y_unstacked, 'h': heights, 'w': widths, 'ser_legend': ser})
        p.rect('x', 'y', source=source, height='h', color=c, fill_alpha=alpha, width='w', line_color=None, line_width=None)
    elif glyph_type =='Area' and y_unstacked != [0]*len(y_unstacked):
        if y_bases is None: y_bases = [0]*len(ys)
        xs_around = xs + xs[::-1]
        ys_around = y_bases + ys[::-1]
        source = bms.ColumnDataSource({'x': [xs_around], 'y': [ys_around], 'x_legend': [wdg['x'].value], 'y_legend': [wdg['y'].value], 'ser_legend': [series]})
        p.patches('x', 'y', source=source, alpha=alpha, fill_color=c, line_color=None, line_width=None)

def create_maps(df, wdg, cols):
    '''
    Create maps based on an input dataframe.The second-to-last column of this
    dataframe is assumed to be the x-axis, or the column of regions that are to be mapped.
    The last column of this dataframe will be the y axis, or the values that correspond
    to the regions. Values are binned into ranges and then mapped to a color for each region.

    Args:
        df (pandas dataframe): input dataframe described above
        wdg (ordered dict): Dictionary of bokeh model widgets.
        cols (dict): Keys are categories of columns of df_source, and values are a list of columns of that category.
    Returns:
        maps (list of bokeh.plotting.figure): These maps are created by the create_map function.
        breakpoints (list of float): Breakpoints that separate the color-shaded bins.
    '''
    print('***Building Maps...')
    maps = []
    breakpoints = []
    regions = ['i','n','r','rnew','rto','st']
    x_axis = df.iloc[:,-2]
    y_axis = df.iloc[:,-1]
    if x_axis.name not in regions or y_axis.dtype == object:
        print('***Error. Did you make sure to set x-axis to a region?')
        return (maps, breakpoints) #empty list
    #find x and y ranges based on the mins and maxes of the regional boundaries for only regions that
    #are in the data
    filepath = this_dir_path + '/in/gis_' + x_axis.name + '.csv'
    region_boundaries = pd.read_csv(filepath, sep=',', dtype={'id': object, 'group': object})
    #Remove holes
    region_boundaries = region_boundaries[region_boundaries['hole'] == False]
    #keep only regions that are in df
    region_boundaries = region_boundaries[region_boundaries['id'].isin(x_axis.unique())]
    #Add x and y columns to region_boundaries and find x and y ranges
    region_boundaries['x'] = region_boundaries['long']*53
    region_boundaries['y'] = region_boundaries['lat']*69
    ranges = {
        'x_max': region_boundaries['x'].max(),
        'x_min': region_boundaries['x'].min(),
        'y_max': region_boundaries['y'].max(),
        'y_min': region_boundaries['y'].min(),
    }
    #set breakpoints depending on the binning strategy
    if wdg['map_bin'].value == 'Auto Equal Num': #an equal number of data ponts in each bin
        map_num_bins = int(wdg['map_num'].value)
        #with full list of values, find uniques with set, and return a sorted list of the uniques
        val_list = sorted(set(y_axis.tolist()))
        #bin indices, find index breakpoints, and convert into value breakpoints.
        index_step = (len(val_list) - 1)/map_num_bins
        indices = [int((i+1)*index_step) for i in range(map_num_bins - 1)]
        breakpoints = [val_list[i] for i in indices]
    elif wdg['map_bin'].value == 'Auto Equal Width': #bins of equal width
        map_num_bins = int(wdg['map_num'].value)
        if wdg['map_min'].value != '' and wdg['map_max'].value != '':
            map_min = float(wdg['map_min'].value)
            map_max = float(wdg['map_max'].value)
            bin_width = (map_max - map_min)/(map_num_bins - 2)
            breakpoints = [map_min + bin_width*i for i in range(map_num_bins - 1)]
        else:
            bin_width = float(y_axis.max() - y_axis.min())/map_num_bins
            map_min = y_axis.min() + bin_width
            map_max = y_axis.max() - bin_width
            breakpoints = [map_min + bin_width*i for i in range(map_num_bins - 1)]
    elif wdg['map_bin'].value == 'Manual':
        breakpoints = [float(bp) for bp in wdg['map_manual'].value.split(',')]

    colors_full = get_map_colors(wdg, breakpoints)

    df_maps = df.copy()
    #assign all y-values to bins
    df_maps['bin_index'] = y_axis.apply(get_map_bin_index, args=(breakpoints,))
    #If there are only 3 columns (x_axis, y_axis, and bin_index), that means we aren't exploding:
    if len(df_maps.columns) == 3:
        maps.append(create_map(df_maps, ranges, region_boundaries, wdg, colors_full))
        print('***Done building map.')
        return (maps, breakpoints) #single map
    #Otherwise we are exploding.
    #find all unique groups of the explode columns.
    df_unique = df_maps.copy()
    #remove x, y, and bin_index
    df_unique = df_unique[df_unique.columns[0:-3]]
    df_unique.drop_duplicates(inplace=True)
    #Loop through rows of df_unique, filter df_maps based on values in each row,
    #and send filtered dataframe to mapping function
    for i, row in df_unique.iterrows():
        reg_bound = region_boundaries
        df_map = df_maps
        title = ''
        for col in df_unique:
            df_map = df_map[df_map[col] == row[col]]
            title = title + col + '=' + str(row[col]) + ', '
        reg_bound = reg_bound[reg_bound['id'].isin(df_map[x_axis.name].unique())]
        #Use filtered regions to set map ranges
        ranges = {
            'x_max': reg_bound['x'].max(),
            'x_min': reg_bound['x'].min(),
            'y_max': reg_bound['y'].max(),
            'y_min': reg_bound['y'].min(),
        }
        #preserve just x axis, y axis, and bin index
        df_map = df_map[df_map.columns[-3:]]
        #remove final comma of title
        title = title[:-2]
        maps.append(create_map(df_map, ranges, reg_bound, wdg, colors_full, title))
    print('***Done building maps.')
    return (maps, breakpoints) #multiple maps

def create_map(df, ranges, region_boundaries, wdg, colors_full, title=''):
    '''
    Create map based on an input dataframe.The third-to-last column of this
    dataframe is assumed to be the column of regions that are to be mapped.
    The second-to-last column is assumed to be the values that are to be mapped.
    The last column is the bin number that determines the color that is applied
    to each region.

    Args:
        df (pandas dataframe): input dataframe described above
        region_boundaries (pandas dataframe): This dataframe has columns for region id, group (if the region has non-contiguous pieces), and x and y values of all boundary points.
        wdg (ordered dict): Dictionary of bokeh model widgets.
        colors_full (list of strings): Colors to shade the map
        title (string): The displayed title for this map
    Returns:
        fig_map (bokeh.plotting.figure): the bokeh figure for the map.
    '''

    df_regions = df.iloc[:,0].tolist()
    df_values = df.iloc[:,1].tolist()
    df_bins = df.iloc[:,2].tolist()

    xs = [] #list of lists of x values of boundaries of regions
    ys = [] #list of lists of y values of boundaries of regions
    regions = []
    values = []
    colors = []
    for grp in region_boundaries['group'].unique().tolist():
        region_boundary = region_boundaries[region_boundaries['group'] == grp]
        xs.append(region_boundary['x'].values.tolist())
        ys.append(region_boundary['y'].values.tolist())
        reg = region_boundary['id'].iloc[0]
        regions.append(reg)
        if reg in df_regions:
            index = df_regions.index(reg)
            value = df_values[index]
            values.append(value)
            bin_num = df_bins[index]
            colors.append(colors_full[int(bin_num)])
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
    width = wdg['map_width'].value
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
    fig_map.title.text_font_size = wdg['map_font_size'].value + 'pt'
    fig_map.toolbar.active_drag = TOOLS[0]
    if wdg['bokeh_tools'].value == 'No':
        fig_map.toolbar.logo = None
        fig_map.toolbar_location = None
    fig_map.grid.grid_line_color = None
    fig_map.patches('x', 'y', source=source, fill_color='color', fill_alpha=float(wdg['map_opacity'].value), line_color="black", line_width=float(wdg['map_line_width'].value))
    return fig_map

def get_map_bin_index(val, breakpoints):
    '''
    Helper function for determining the bin number for a given value and set of breakpoints.
    This assumes that bin ranges are less than or equal to the upper value and
    strictly greater than the lower value.

    Args:
        val (float): The value that is to be binned
        breakpoints (list of float): Breakpoints that separate the color-shaded bins.
    Returns:
        bin index (int): the bin number that will determine the color of the region.
    '''
    for i, breakpoint in enumerate(breakpoints):
        if val <= breakpoint:
            return i
    return len(breakpoints)


def build_map_legend(wdg, breakpoints):
    '''
    Return html for map legend, based on supplied labels and global COLORS

    Args:
        wdg (ordered dict): Dictionary of bokeh model widgets.
        breakpoints (list of float): Breakpoints that separate the color-shaded bins.
    Returns:
        legend_string (string): full html to be used as legend.
    '''

    if wdg['map_bin'].value == 'Manual':
        breakpoint_strings = [str(bp) for bp in breakpoints]
    else:
        breakpoint_strings = prettify_numbers(breakpoints)

    labels = ['<= ' + breakpoint_strings[0]]
    labels += [breakpoint_strings[i] + ' - ' + breakpoint_strings[i+1] for i in range(len(breakpoint_strings) - 1)]
    labels += ['> ' + breakpoint_strings[-1]]

    colors = get_map_colors(wdg, breakpoints)
    legend_string = build_legend(labels, colors)
    return legend_string

def get_map_colors(wdg, breakpoints):
    '''
    Return list of map colors. If we have a second palette, we will reverse it and use it
    at the beginning. We also may have map_palette_break, which will be used as a breakpoint
    to separate the colors.

    Args:
        wdg (ordered dict): Dictionary of bokeh model widgets.
        breakpoints (list of float): Breakpoints that separate the color-shaded bins.
    Returns:
        List of hex strings that represent map colors
    '''

    palette = wdg['map_palette'].value
    palette_2 = wdg['map_palette_2'].value
    palette_break = wdg['map_palette_break'].value
    num = len(breakpoints) + 1

    if palette_2 == '':
        return get_palette(palette, num)
    else:
        if palette_break == '':
            if num % 2 == 0:
                #even number of bins, so we split num equally between the palettes
                return list(reversed(get_palette(palette_2, num/2))) + get_palette(palette, num/2)
            else:
                #odd number of bins, so the middle bin is white and we split the rest equally between the other palettes
                return list(reversed(get_palette(palette_2, (num-1)/2))) + ['#ffffff'] + get_palette(palette, (num-1)/2)
        else:
            palette_break = float(palette_break)
            bp_low = [bp for bp in breakpoints if bp < palette_break]
            num_low = len(bp_low)
            num_high = num - num_low - 1
            if palette_break in breakpoints:
                return list(reversed(get_palette(palette_2, num_low + 1))) + get_palette(palette, num_high)
            else:
                return list(reversed(get_palette(palette_2, num_low))) + ['#ffffff'] + get_palette(palette, num_high)


def get_palette(palette, num):
    '''
    Return list of colors in palette based on the palette name and number of bins.
    Most are taken from bokeh.palettes.all_palettes[palette][num], but 'all_red', 'all_green',
    'all_blue', and 'all_gray' are also allowed as names for palette, and are generated from
    bokeh.palettes.linear_palette().

    Args:
        palette (string): The name of the palette
        num (int): Number of colors to return
    Returns:
        List of hex strings in palette
    '''

    if palette.startswith('all_'):
        if palette == 'all_red':
            return bpa.linear_palette(['#%02x%02x%02x' % (255, 255-i, 255-i) for i in range(256)],num)
        elif palette == 'all_green':
            return bpa.linear_palette(['#%02x%02x%02x' % (255-i, 255, 255-i) for i in range(256)],num)
        elif palette == 'all_blue':
            return bpa.linear_palette(['#%02x%02x%02x' % (255-i, 255-i, 255) for i in range(256)],num)
        elif palette == 'all_gray':
            return bpa.linear_palette(['#%02x%02x%02x' % (255-i, 255-i, 255-i) for i in range(256)],num)
    else:
        return list(reversed(bpa.all_palettes[palette][num]))

def build_plot_legend(df_plots, series_val, custom_colors):
    '''
    Return html for series legend, based on values of column that was chosen for series, and global COLORS.

    Args:
        df_plots (pandas dataframe): Dataframe of all plots data.
        series_val (string): Header for column chosen as series.
        custom_colors (dict): Keys are column names and values are dicts that map column values to colors (hex strings)

    Returns:
        legend_string (string): html to be used as legend.
    '''
    if series_val == 'None':
        return ''
    labels = df_plots[series_val].unique().tolist()
    colors = [COLORS[i] for i, t in enumerate(labels)]
    if custom_colors and series_val in custom_colors:
        for i, lab in enumerate(labels):
             if lab in custom_colors[series_val]:
                 colors[i] = custom_colors[series_val][lab]
    labels.reverse()
    colors.reverse()
    legend_string = build_legend(labels, colors)
    return legend_string

def build_legend(labels, colors):
    '''
    Return html for legend, based on list of labels and list of colors

    Args:
        labels(list of strings): Displayed labels for each legend entry
        colors (list of strings): List of color strings using hexidecimal format
    Returns:
        legend_string (string): html to be used as legend.
    '''
    legend_string = ''
    for i, txt in enumerate(labels):
        legend_string += '<div class="legend-entry"><span class="legend-color" style="background-color:' + str(colors[i]) + ';"></span>'
        legend_string += '<span class="legend-text">' + str(txt) +'</span></div>'
    return legend_string

def display_config(wdg, wdg_defaults):
    '''
    Build HTML for displaying list of non-default widget configurations

    Args:
        wdg (ordered dict): Dictionary of bokeh model widgets.
        wdg_defaults (dict): Keys are widget names and values are the default values of the widgets.

    Returns:
        output: HTML of displayed list of widget configurations
    '''
    output = '<div class="config-display-title">Config Summary</div>'
    for key in wdg_defaults:
        if key not in ['data', 'chart_type']:
            label = key
            item_string = False
            if isinstance(wdg[key], bmw.groups.Group) and wdg[key].active != wdg_defaults[key]:
                if key.startswith('filter_'):
                    label = 'filter-' + wdg['heading_'+key].text
                item_string = ''
                active_indices = wdg[key].active
                for i in active_indices:
                    item_string += wdg[key].labels[i] + ', '
            elif isinstance(wdg[key], bmw.inputs.InputWidget) and wdg[key].value != wdg_defaults[key]:
                item_string = wdg[key].value
            if item_string != False:
                if item_string == '':
                    item_string = 'NONE'
                output += '<div class="config-display-item"><span class="config-display-key">' + label + ': </span>' + item_string + '</div>'
    return output

def apply_aggregation(group, agg_method, y_col, wdg_range, kw):
    """
    Helper function for pandas dataframe groupby object with apply function.

    Args:
        agg_method (string): The aggregation method to apply
        group (pandas dataframe): This has columns required for weighted average
        y_col (string): Name of the column for which an aggregation is calculated
        chart_type (string): The type of chart to be built. If this is a Range chart, then we need to add the range_min and range_max data.
        kw (dict): Keyword arguments
            y_weight (string): Name of column that will be used as weighting factor for Weighted Ave
            y_weight_numer (string): Name of column that will be used as weighting factor for numerator of Weighted Ave Ratio
            y_weight_denom (string): Name of column that will be used as weighting factor for denominator of Weighted Ave Ratio
    Returns:
        weighted average (float): The weighted average using the two specified columns
    """
    d = group[y_col]
    agg_result = None
    try:
        if agg_method == 'Sum':
            agg_result = d.sum()
        elif agg_method == 'Ave':
            agg_result = d.mean()
        elif agg_method == 'Weighted Ave' and 'y_weight' in kw:
            w = group[kw['y_weight']]
            agg_result = (d * w).sum() / w.sum()
        elif agg_method == 'Weighted Ave Ratio' and 'y_weight_numer' in kw and 'y_weight_denom' in kw:
            wn = group[kw['y_weight_numer']]
            wd = group[kw['y_weight_denom']]
            agg_result = ((d * wn).sum() / wn.sum())/((d * wd).sum() / wd.sum())
    except ZeroDivisionError:
        return pd.DataFrame({y_col: [None]})
    if wdg_range == 'Within Series':
        return pd.DataFrame({y_col: [agg_result], 'range_min': [d.min()], 'range_max': [d.max()]})
    else:
        return pd.DataFrame({y_col: [agg_result]})

def op_with_base(group, op, col, col_base, y_val):
    """
    Helper function for pandas dataframe groupby object with apply function. This returns a pandas
    dataframe with an operation applied to one of the columns.

    Args:
        group (pandas dataframe): This has columns required for performing the operation
        op (string): The type of operation: 'Difference', 'Ratio'
        col (string): The column across which the operation is happening
        col_base (string): The value of col to be used as the base for the operation, or "Consecutive" or "Total"
        y_val (string): Name of column that will be modified according to the operation.
    Returns:
        group_out (pandas dataframe): A like-indexed dataframe with the specified operations.
    """
    group_out = group.copy()
    if col_base == 'Consecutive':
        if op == 'Difference':
            group_out[y_val] = group[y_val] - group[y_val].shift()
        elif op == 'Ratio':
            group_out[y_val] = group[y_val] / group[y_val].shift()
    elif col_base == 'Total':
        if op == 'Difference':
            group_out[y_val] = group[y_val] - group[y_val].sum()
        elif op == 'Ratio':
            group_out[y_val] = group[y_val] / group[y_val].sum()
    else:
        df_base = group[group[col]==col_base]
        if df_base.empty:
            y_base = 0
        else:
            y_base = df_base[y_val].iloc[0]
        if op == 'Difference':
            group_out[y_val] = group[y_val] - y_base
        elif op == 'Ratio':
            group_out[y_val] = group[y_val] / y_base if y_base else 0
    return group_out

def prettify_numbers(number_list):
    str_list = []
    for x in number_list:
        if abs(x) < .001 or abs(x) >= 100000:
            s = '%.2E' % x
        else:
            s = round_to_n(x, 3)
        s = str(s)
        if s.endswith('.0'):
            s = s[:-2]
        str_list.append(s)
    return str_list

def round_to_n(x, n):
    #https://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
    return round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))

def update_data(attr, old, new):
    '''
    When data source is updated, call update_data_source()
    '''
    update_data_source()

def update_data_source(init_load=False, init_config={}):
    '''
    When data source is updated (or on initial load), update the widgets
    section of the layout based on if the input path is a csv, gdx, or ReEDS result.

    Args:
        init_load (boolean): True if this is the initial load of the page
        init_config (dict): Initial configuration supplied by the URL.
    Returns:
        Nothing: All plots are cleared, and widgets are set to accept further configuration.
    '''
    GL['widgets'] = GL['data_source_wdg'].copy()
    GL['wdg_defaults'] = collections.OrderedDict()
    GL['wdg_defaults']['data'] = ''
    path = GL['data_source_wdg']['data'].value
    path = path.replace('"', '')
    if path == '':
        pass
    elif path.lower().endswith('.csv') and not path.lower().endswith('reeds_scenarios.csv'):
        GL['df_source'], GL['columns'] = get_df_csv(path)
        GL['widgets'].update(build_widgets(GL['df_source'], GL['columns'], init_load, init_config, wdg_defaults=GL['wdg_defaults']))
    elif path.lower().endswith('.gdx'):
        GL['widgets'].update(get_wdg_gdx(path, GL['widgets']))
    else: #reeds path was entered
        rb.update_reeds_data_source(path, init_load, init_config)
    GL['controls'].children = list(GL['widgets'].values())
    GL['plots'].children = []

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
    #show widget config
    GL['widgets']['display_config'].text = display_config(GL['widgets'], GL['wdg_defaults'])
    
    if GL['widgets']['x'].value == 'None' or GL['widgets']['y'].value == 'None':
        GL['plots'].children = []
        return
    GL['df_plots'] = set_df_plots(GL['df_source'], GL['columns'], GL['widgets'], GL['custom_sorts'])
    if GL['widgets']['render_plots'].value == 'Yes':
        if GL['widgets']['chart_type'].value == 'Map':
            figs, breakpoints = create_maps(GL['df_plots'], GL['widgets'], GL['columns'])
            legend_text = build_map_legend(GL['widgets'], breakpoints)
        else:
            figs = create_figures(GL['df_plots'], GL['widgets'], GL['columns'], GL['custom_colors'])
            legend_text = build_plot_legend(GL['df_plots'], GL['widgets']['series'].value, GL['custom_colors'])
        GL['widgets']['legend'].text = legend_text
        GL['plots'].children = figs

def download_url(dir_path='', auto_open=True):
    '''
    Export configuration URL query string and python dict formats to text file
    '''
    wdg = GL['widgets']
    wdg_defaults = GL['wdg_defaults']
    non_defaults = {}
    for key in wdg_defaults:
        if isinstance(wdg[key], bmw.groups.Group) and wdg[key].active != wdg_defaults[key]:
            non_defaults[key] = wdg[key].active
        elif isinstance(wdg[key], bmw.inputs.InputWidget) and wdg[key].value != wdg_defaults[key] and key not in ['auto_update','presets','report_options','report_custom','report_base']:
            non_defaults[key] = wdg[key].value
    json_string = json.dumps(non_defaults)
    url_query = '?widgets=' + urlp.quote(json_string)
    if dir_path == '':
        path = user_out_path + '/url-'+ datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")+'.txt'
    else:
        path = dir_path + '/url.txt'
    with open(path, 'w') as f:
        f.write('Paste this after "/bokehpivot" in the URL:\n' + url_query + '\n\n')
    if auto_open:
        sp.Popen(os.path.abspath(path), shell=True)

def download_config(dir_path, auto_open, format):
    '''
    Export report configuration to python file
    '''
    wdg = GL['widgets']
    wdg_defaults = GL['wdg_defaults']
    non_defaults = {}
    if format == 'report':
        config_string = "{'name': 'Section Name', 'config': {"
    elif format == 'reeds_preset':
        config_string = "('Preset Name', {"
    filter_string = "'filter': {"
    for key in wdg_defaults:
        if isinstance(wdg[key], bmw.groups.Group) and wdg[key].active != wdg_defaults[key] and key not in ['scenario_filter']:
            if key.startswith('filter_'):
                title = wdg['heading_'+key].text
                labels = ["'" + wdg[key].labels[i] + "'" for i in wdg[key].active]
                filter_string += "'" + title + "':[" + ",".join(labels) + "], "
            else:
                labels = ["'" + wdg[key].labels[i] + "'" for i in wdg[key].active]
                config_string += "'" + key + "':[" + ",".join(labels) + "], "
        elif isinstance(wdg[key], bmw.inputs.InputWidget) and wdg[key].value != wdg_defaults[key] and key not in ['data','auto_update','presets','report_options','report_custom','report_base']:
            raw_flag = ''
            if isinstance(wdg[key], bmw.inputs.TextInput):
                raw_flag = 'r'
            if key != 'result' or format != 'reeds_preset':
                config_string += "'" + key + "':" + raw_flag + "'" + str(wdg[key].value) + "', "

    if format == 'report':
        config_string += filter_string + '}}},'
        file_name = 'report'
    elif format == 'reeds_preset':
        config_string += filter_string + '}}),'
        file_name = 'preset'
    if dir_path == '':
        path = user_out_path + '/' + file_name + '-'+ datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")+'.py'
    else:
        path = dir_path + '/' + file_name + '.py'
    with open(path, 'w') as f:
        if format == 'report':
            f.write('static_presets = [\n' + config_string + '\n]\n')
            f.write('#This file may be used directly as a custom report in the Build Report section of the interface, or the section above may be added to another custom report.\n')
        elif format == 'reeds_preset':
            f.write(config_string + '\n')
            f.write('#Add this to the "presets" section of the appropriate result in reeds.py.\n')
    if auto_open:
        sp.Popen(os.path.abspath(path), shell=True)

def download_report(dir_path='', auto_open=True):
    download_config(dir_path, auto_open, 'report')

def download_reeds_preset(dir_path='', auto_open=True):
    download_config(dir_path, auto_open, 'reeds_preset')

def download_csv(dir_path='', auto_open=True):
    '''
    Download a csv file of the currently viewed data to the downloads/ directory,
    with the current timestamp.
    '''
    print('***Downloading View...')
    if dir_path == '':
        path = user_out_path + '/view-'+ datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")+'.csv'
    else:
        path = dir_path + '/view.csv'
    GL['df_plots'].to_csv(path, index=False)
    print('***Done downloading View to ' + path)
    if auto_open:
        sp.Popen(os.path.abspath(path), shell=True)

def download_html(dir_path='', auto_open=True):
    '''
    Download html file of the currently viewed data to the downloads/ directory,
    with the current timestamp.
    '''
    print('***Downloading View...')
    if dir_path == '':
        html_path = user_out_path + '/view-'+ datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")+'.html'
    else:
        html_path = dir_path + '/view.html'
    static_plots = []
    legend = bmw.Div(text=GL['widgets']['legend'].text)
    plots = GL['plots'].children
    GL['plots'].children = []
    static_plots.append(bl.row(plots + [legend]))
    with open(this_dir_path + '/templates/static/index.html', 'r') as template_file:
        template_string=template_file.read()
    template = ji.Template(template_string)
    resources = br.Resources()
    html = be.file_html(static_plots, resources=resources, template=template)
    with open(html_path, 'w') as f:
        f.write(html)
    if auto_open:
        sp.Popen(os.path.abspath(html_path), shell=True)
    GL['plots'].children = plots
    print('***Done downloading View to ' + html_path)

def download_all():
    '''
    Download all of the outputs of a view into a timestamped folder
    '''
    dir_path = user_out_path + '/view-'+ datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    os.makedirs(dir_path, False)
    download_html(dir_path, False)
    download_csv(dir_path, False)
    download_url(dir_path, False)
    download_report(dir_path, False)
    download_reeds_preset(dir_path, False)

def download_source(dir_path='', auto_open=True):
    '''
    Download a csv file of the full data source to the downloads/ directory,
    with the current timestamp.
    '''
    print('***Downloading full source...')
    if dir_path == '':
        path = user_out_path + '/source-'+ datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")+'.csv'
    else:
        path = dir_path + '/source.csv'
    GL['df_source'].to_csv(path, index=False)
    print('***Done downloading full source to ' + path)
    if auto_open:
        sp.Popen(os.path.abspath(path), shell=True)
