'''
ReEDS 2.0 results metadata and preprocess functions.

When adding a new ReEDS result and associated presets, this should be the only file you need to modify.

There are three sections:
1. Preprocess functions: Only needed for a result if the gdx data needs to be manipulated
2. Columns metatdata: This allows column values from a result to be mapped to display categories, joined with other columns, and styled
3. Results metadata: This is where all result configuration happens
'''
from __future__ import division
import os
import pandas as pd
import collections
import core

rb_globs = {'output_subdir': '\\outputs\\', 'test_file': 'cap.csv', 'report_subdir':'/reeds2'}
this_dir_path = os.path.dirname(os.path.realpath(__file__))
CRF_reeds = 0.077
df_deflator = pd.read_csv(this_dir_path + '/in/inflation.csv', index_col=0)
ILR_UPV = 1.3
ILR_distPV = 1.1
costs_orig_inv = ['orig_inv_investment_capacity_costs','orig_inv_investment_refurbishment_capacity']
costs_pol_inv = ['inv_investment_capacity_costs','inv_investment_refurbishment_capacity','inv_ptc_payments_negative','inv_ptc_payments_negative_refurbishments']

#1. Preprocess functions for results_meta
def scale_column(df, **kw):
    df[kw['column']] = df[kw['column']] * kw['scale_factor']
    return df

def scale_column_filtered(df, **kw):
    cond = df[kw['by_column']].isin(kw['by_vals'])
    df.loc[cond, kw['change_column']] = df.loc[cond, kw['change_column']] * kw['scale_factor']
    return df

def sum_over_cols(df, **kw):
    df = df.drop(kw['sum_over_cols'], axis='columns')
    df =  df.groupby(kw['group_cols'], sort=False, as_index =False).sum()
    return df

def apply_inflation(df, **kw):
    df[kw['column']] = inflate_series(df[kw['column']])
    return df

def inflate_series(ser_in):
    return ser_in * 1/df_deflator.loc[int(core.GL['widgets']['var_dollar_year'].value),'Deflator']

def discount_costs_bulk(df, **kw):
    d = 0.069456772
    y0 = int(core.GL['widgets']['var_pv_year'].value)
    df['Discounted Cost (Bil $)'] = df['Cost (Bil $)'] / (1 + d)**(df['year'] - y0)
    return df

def discount_costs(df, **kw):
    #inner join the cost_cat_type.csv table to get types of costs (Capital, Operation)
    cost_cat_type = pd.read_csv(this_dir_path + '/in/reeds2/cost_cat_type.csv')
    df = pd.merge(left=df, right=cost_cat_type, on='cost_cat', sort=False)
    #make new column that is the pv multiplier
    df['pv_mult'] = df.apply(lambda x: get_pv_mult(int(x['year']), x['type']), axis=1)
    df['Discounted Cost (Bil $)'] = df['Cost (Bil $)'] * df['pv_mult']
    return df

#Return present value multiplier
def get_pv_mult(year, type, dinvest=0.054439024, dsocial=0.03, lifetime=20):
    refyear = int(core.GL['widgets']['var_pv_year'].value)
    lastyear = int(core.GL['widgets']['var_end_year'].value)
    if type == "Operation":
        pv_mult = 1 / (1 + dsocial)**(year - refyear)
    elif type == "Capital":
        pv_mult = CRF(dinvest, lifetime) / CRF(dinvest, min(lifetime, lastyear + 1 - year)) * 1 / (1 + dsocial)**(year - refyear)
    return pv_mult

#Capital recovery factor
def CRF(i,n):
    tempn = n
    if tempn == 0:
        tempn = 1
        print('Data goes beyond Present Value End Year. Filter out data beyond this year for proper system cost calculation.')
    return i/(1-(1/(1+i)**tempn))

def map_i_to_n(df, **kw):
    df_hier = pd.read_csv(this_dir_path + '/in/reeds2/region_map.csv')
    dict_hier = dict(zip(df_hier['s'], df_hier['n']))
    df.loc[df['region'].isin(dict_hier.keys()), 'region'] = df['region'].map(dict_hier)
    df.rename(columns={'region': 'n'}, inplace=True)
    return df

def remove_n(df, **kw):
    df = df[~df['region'].astype(str).str.startswith('p')].copy()
    df['region'] = df['region'].map(lambda x: x.lstrip('s'))
    df.rename(columns={'region': 'i'}, inplace=True)
    return df

def pre_val_streams(df, **kw):
    df_not_dol = df[df['con_name'].isin(['mwh','kw'])].copy()
    df_dol = df[~df['con_name'].isin(['mwh','kw'])].copy()
    #apply inflation and annualize
    df_dol['value'] = inflate_series(df_dol['value']) * CRF_reeds
    #adjust capacity of PV???
    df = pd.concat([df_not_dol, df_dol],sort=False,ignore_index=True)
    return df

def pre_reduced_cost(df, **kw):
    df['icrb'] = df['tech'] + ' | ' + df['vintage'] + ' | ' + df['region'] + ' | ' + df['bin']
    return df

def pre_lcoe(dfs, **kw):
    #Apply inflation
    dfs['lcoe']['$/MWh'] = inflate_series(dfs['lcoe']['$/MWh'])
    #Merge with available capacity
    df = pd.merge(left=dfs['lcoe'], right=dfs['avail'], how='left', on=['tech', 'region', 'year', 'bin'], sort=False)
    df['available MW'].fillna(0, inplace=True)
    df['available'] = 'no'
    df.loc[df['available MW'] > 0.001, 'available'] = 'yes'
    #Merge with chosen capacity
    df = pd.merge(left=df, right=dfs['inv'], how='left', on=['tech', 'vintage', 'region', 'year', 'bin'], sort=False)
    df['chosen MW'].fillna(0, inplace=True)
    df['chosen'] = 'no'
    df.loc[df['chosen MW'] != 0, 'chosen'] = 'yes'
    #Add icrb column
    df['icrb'] = df['tech'] + ' | ' + df['vintage'] + ' | ' + df['region'] + ' | ' + df['bin']
    return df

#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------

#2. Columns metadata. These are columns that are referenced in the Results section below.
#This is where joins, maps, and styles are applied for the columns.
#For 'style', colors are in hex, but descriptions are given (see http://www.color-hex.com/color-names.html).
columns_meta = {
    'tech':{
        'type': 'string',
        'map': this_dir_path + '/in/reeds2/tech_map.csv',
        'style': this_dir_path + '/in/reeds2/tech_style.csv',
    },
    'class':{
        'type': 'string',
    },
    'region':{
        'type': 'string',
    },
    'i':{
        'type': 'string',
        'join': this_dir_path + '/in/hierarchy.csv',
    },
    'n':{
        'type': 'string',
        'join': this_dir_path + '/in/hierarchy.csv',
    },
    'timeslice':{
        'type': 'string',
        'map': this_dir_path + '/in/m_map.csv',
        'style': this_dir_path + '/in/m_style.csv',
    },
    'year':{
        'type': 'number',
        'filterable': True,
        'seriesable': True,
        'y-allow': False,
    },
    'icrb':{
        'type': 'string',
        'filterable': False,
        'seriesable': False,
    },
    'cost_cat':{
        'type': 'string',
        'map': this_dir_path + '/in/reeds2/cost_cat_map.csv',
        'style': this_dir_path + '/in/reeds2/cost_cat_style.csv',
    },
}

#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------

#3. Results metadata. This is where all ReEDS results are defined. Parameters are read from gdx files, and
#are converted into pandas dataframes for pivoting. Preprocess functions may be used to perform additional manipulation.
#Note that multiple parameters may be read in for the same result (search below for 'sources')
#Presets may also be defined.
results_meta = collections.OrderedDict((
    ('Capacity BA (GW)',
        {'file': 'cap.csv',
        'columns': ['tech', 'region', 'year', 'Capacity (GW)'],
        'preprocess': [
            {'func': map_i_to_n, 'args': {}},
            {'func': scale_column, 'args': {'scale_factor': .001, 'column': 'Capacity (GW)'}},
        ],
        'index': ['tech', 'n', 'year'],
        'presets': collections.OrderedDict((
            ('Stacked Area',{'x':'year','y':'Capacity (GW)','series':'tech', 'explode': 'scenario','chart_type':'Area'}),
            ('Stacked Bars',{'x':'year','y':'Capacity (GW)','series':'tech', 'explode': 'scenario','chart_type':'Bar', 'bar_width':'1.75'}),
            ('Explode By Tech',{'x':'year','y':'Capacity (GW)','series':'scenario', 'explode': 'tech','chart_type':'Line'}),
            ('PCA Map Final by Tech',{'x':'n','y':'Capacity (GW)', 'explode': 'scenario','explode_group': 'tech','chart_type':'Map', 'filter': {'year': 'last'}}),
            ('State Map Final by Tech',{'x':'st','y':'Capacity (GW)', 'explode': 'scenario','explode_group': 'tech','chart_type':'Map', 'filter': {'year': 'last'}}),
        )),
        }
    ),

    ('New Annual Capacity BA (GW)',
        {'file': 'cap_new_ann.csv',
        'columns': ['tech', 'region', 'year', 'Capacity (GW)'],
        'preprocess': [
            {'func': map_i_to_n, 'args': {}},
            {'func': scale_column, 'args': {'scale_factor': .001, 'column': 'Capacity (GW)'}},
        ],
        'index': ['tech', 'n', 'year'],
        'presets': collections.OrderedDict((
            ('Stacked Area',{'x':'year','y':'Capacity (GW)','series':'tech', 'explode': 'scenario','chart_type':'Area'}),
            ('Stacked Bars',{'x':'year','y':'Capacity (GW)','series':'tech', 'explode': 'scenario','chart_type':'Bar', 'bar_width':'1.75'}),
            ('Explode By Tech',{'x':'year','y':'Capacity (GW)','series':'scenario', 'explode': 'tech','chart_type':'Line'}),
            ('PCA Map Final by Tech',{'x':'n','y':'Capacity (GW)', 'explode': 'scenario','explode_group': 'tech','chart_type':'Map', 'filter': {'year': 'last'}}),
            ('State Map Final by Tech',{'x':'st','y':'Capacity (GW)', 'explode': 'scenario','explode_group': 'tech','chart_type':'Map', 'filter': {'year': 'last'}}),
        )),
        }
    ),

    ('Annual Retirements BA (GW)',
        {'file': 'ret_ann.csv',
        'columns': ['tech', 'region', 'year', 'Capacity (GW)'],
        'preprocess': [
            {'func': map_i_to_n, 'args': {}},
            {'func': scale_column, 'args': {'scale_factor': .001, 'column': 'Capacity (GW)'}},
        ],
        'index': ['tech', 'n', 'year'],
        'presets': collections.OrderedDict((
            ('Stacked Area',{'x':'year','y':'Capacity (GW)','series':'tech', 'explode': 'scenario','chart_type':'Area'}),
            ('Stacked Bars',{'x':'year','y':'Capacity (GW)','series':'tech', 'explode': 'scenario','chart_type':'Bar', 'bar_width':'1.75'}),
            ('Explode By Tech',{'x':'year','y':'Capacity (GW)','series':'scenario', 'explode': 'tech','chart_type':'Line'}),
            ('PCA Map Final by Tech',{'x':'n','y':'Capacity (GW)', 'explode': 'scenario','explode_group': 'tech','chart_type':'Map', 'filter': {'year': 'last'}}),
            ('State Map Final by Tech',{'x':'st','y':'Capacity (GW)', 'explode': 'scenario','explode_group': 'tech','chart_type':'Map', 'filter': {'year': 'last'}}),
        )),
        }
    ),

    ('Capacity Resource Region (GW)',
        {'file': 'cap.csv',
        'columns': ['tech', 'region', 'year', 'Capacity (GW)'],
        'preprocess': [
            {'func': remove_n, 'args': {}},
            {'func': scale_column, 'args': {'scale_factor': .001, 'column': 'Capacity (GW)'}},
        ],
        'index': ['tech', 'i', 'year'],
        'presets': collections.OrderedDict((
            ('Stacked Area',{'x':'year','y':'Capacity (GW)','series':'tech', 'explode': 'scenario','chart_type':'Area'}),
            ('Stacked Bars',{'x':'year','y':'Capacity (GW)','series':'tech', 'explode': 'scenario','chart_type':'Bar', 'bar_width':'1.75'}),
            ('Explode By Tech',{'x':'year','y':'Capacity (GW)','series':'scenario', 'explode': 'tech','chart_type':'Line'}),
            ('RR Map Final by Tech',{'x':'i','y':'Capacity (GW)', 'explode': 'scenario','explode_group': 'tech','chart_type':'Map', 'filter': {'year': 'last'}}),
            ('RR Map Final Wind',{'x':'i','y':'Capacity (GW)', 'explode': 'scenario','chart_type':'Map', 'filter': {'year': 'last', 'tech':['wind-ons','wind-ofs']}}),
        )),
        }
    ),


    ('Generation BA (TWh)',
        {'file': 'gen_ann.csv',
        'columns': ['tech', 'region', 'year', 'Generation (TWh)'],
        'preprocess': [
            {'func': map_i_to_n, 'args': {}},
            {'func': scale_column, 'args': {'scale_factor': 1e-6, 'column': 'Generation (TWh)'}},
        ],
        'index': ['tech', 'n', 'year'],
        'presets': collections.OrderedDict((
            ('Stacked Area',{'x':'year','y':'Generation (TWh)','series':'tech', 'explode': 'scenario','chart_type':'Area'}),
            ('Stacked Bars',{'x':'year','y':'Generation (TWh)','series':'tech', 'explode': 'scenario','chart_type':'Bar', 'bar_width':'1.75'}),
            ('Explode By Tech',{'x':'year','y':'Generation (TWh)','series':'scenario', 'explode': 'tech','chart_type':'Line'}),
            ('PCA Map Final by Tech',{'x':'n','y':'Generation (TWh)', 'explode': 'scenario','explode_group': 'tech','chart_type':'Map', 'filter': {'year': 'last'}}),
            ('State Map Final by Tech',{'x':'st','y':'Generation (TWh)', 'explode': 'scenario','explode_group': 'tech','chart_type':'Map', 'filter': {'year': 'last'}}),
        )),
        }
    ),

    ('Gen by timeslice national (GW)',
        {'file': 'gen_h.csv',
        'columns': ['tech', 'region', 'timeslice', 'year', 'Generation (GW)'],
        'index': ['tech', 'year', 'timeslice'],
        'preprocess': [
            {'func': map_i_to_n, 'args': {}},
            {'func': sum_over_cols, 'args': {'sum_over_cols': ['n'], 'group_cols': ['tech', 'year', 'timeslice']}},
            {'func': scale_column, 'args': {'scale_factor': .001, 'column': 'Generation (GW)'}},
        ],
        'presets': collections.OrderedDict((
            ('Stacked Bars Final',{'x':'timeslice','y':'Generation (GW)','series':'tech', 'explode': 'scenario','chart_type':'Bar', 'filter': {'year': 'last'}}),
        )),
        }
    ),

    ('CO2 Emissions National (MMton)',
        {'file': 'emit_nat.csv',
        'columns': ['year', 'CO2 (MMton)'],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': 1e-6, 'column': 'CO2 (MMton)'}},
        ],
        'index': ['year'],
        'presets': collections.OrderedDict((
            ('Scenario Lines Over Time',{'x':'year','y':'CO2 (MMton)','series': 'scenario','chart_type':'Line'}),
        )),
        }
    ),

    ('CO2 Emissions BA (MMton)',
        {'file': 'emit_r.csv',
        'columns': ['n', 'year', 'CO2 (MMton)'],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': 1e-6, 'column': 'CO2 (MMton)'}},
        ],
        'index': ['n','year'],
        'presets': collections.OrderedDict((
            ('Final BA Map',{'x':'n','y':'CO2 (MMton)','explode': 'scenario','chart_type':'Map', 'filter': {'year': 'last'}}),
        )),
        }
    ),

    ('Energy Price National ($/MWh)',
        {'file': 'price_nat.csv',
        'columns': ['year', '$/MWh'],
        'preprocess': [
            {'func': apply_inflation, 'args': {'column': '$/MWh'}},
        ],
        'index': ['year'],
        'presets': collections.OrderedDict((
            ('Scenario Lines Over Time',{'x':'year','y':'$/MWh','series': 'scenario','chart_type':'Line'}),
        )),
        }
    ),

    ('Energy Price BA ($/MWh)',
        {'file': 'price_ann.csv',
        'columns': ['n', 'year', '$/MWh'],
        'preprocess': [
            {'func': apply_inflation, 'args': {'column': '$/MWh'}},
        ],
        'index': ['n', 'year'],
        'presets': collections.OrderedDict((
            ('Final BA Map',{'x':'n','y':'$/MWh','explode': 'scenario','chart_type':'Map', 'filter': {'year': 'last'}}),
        )),
        }
    ),

    ('Sys Cost (Bil $)',
        {'file': 'systemcost.csv',
        'columns': ['cost_cat', 'year', 'Cost (Bil $)'],
        'index': ['cost_cat', 'year'],
        'preprocess': [
            {'func': apply_inflation, 'args': {'column': 'Cost (Bil $)'}},
            {'func': scale_column, 'args': {'scale_factor': 1e-9, 'column': 'Cost (Bil $)'}},
            {'func': discount_costs, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Stacked Bars',{'x':'scenario','y':'Discounted Cost (Bil $)','series':'cost_cat','chart_type':'Bar'}),
            ('2018-end Stacked Bars',{'x':'scenario','y':'Discounted Cost (Bil $)','series':'cost_cat','chart_type':'Bar', 'filter': {'year': {'start':2018}}}),
        )),
        }
    ),

    ('Sys Cost Bulk (Bil $)',
        {'file': 'systemcost_bulk.csv',
        'columns': ['cost_cat_bulk', 'year', 'Cost (Bil $)'],
        'index': ['cost_cat_bulk', 'year'],
        'preprocess': [
            {'func': apply_inflation, 'args': {'column': 'Cost (Bil $)'}},
            {'func': scale_column, 'args': {'scale_factor': 1e-9, 'column': 'Cost (Bil $)'}},
            {'func': discount_costs_bulk, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Total Discounted',{'x':'scenario','y':'Discounted Cost (Bil $)','series':'cost_cat_bulk','chart_type':'Bar', 'filter': {'cost_cat_bulk':{'exclude':costs_orig_inv}}}),
            ('Total Discounted No Pol',{'x':'scenario','y':'Discounted Cost (Bil $)','series':'cost_cat_bulk','chart_type':'Bar', 'filter': {'cost_cat_bulk':{'exclude':costs_pol_inv}}}),
            ('2018-end Discounted',{'x':'scenario','y':'Discounted Cost (Bil $)','series':'cost_cat_bulk','chart_type':'Bar', 'filter': {'cost_cat_bulk':{'exclude':costs_orig_inv}, 'year': {'start':2018}}}),
            ('Discounted by Year',{'x':'year','y':'Discounted Cost (Bil $)','series':'cost_cat_bulk','explode':'scenario','chart_type':'Bar', 'bar_width':'1.75', 'filter': {'cost_cat_bulk':{'exclude':costs_orig_inv}}}),
            ('Undiscounted by Year',{'x':'year','y':'Cost (Bil $)','series':'cost_cat_bulk','explode':'scenario','chart_type':'Bar', 'bar_width':'1.75', 'filter': {'cost_cat_bulk':{'exclude':costs_orig_inv}}}),
        )),
        }
    ),

    ('Sys Cost Bulk EW (Bil $)',
        {'file': 'systemcost_bulk_ew.csv',
        'columns': ['cost_cat_bulk', 'year', 'Cost (Bil $)'],
        'index': ['cost_cat_bulk', 'year'],
        'preprocess': [
            {'func': apply_inflation, 'args': {'column': 'Cost (Bil $)'}},
            {'func': scale_column, 'args': {'scale_factor': 1e-9, 'column': 'Cost (Bil $)'}},
            {'func': discount_costs_bulk, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Total Discounted',{'x':'scenario','y':'Discounted Cost (Bil $)','series':'cost_cat_bulk','chart_type':'Bar', 'filter': {'cost_cat_bulk':{'exclude':costs_orig_inv}}}),
            ('Total Discounted No Pol',{'x':'scenario','y':'Discounted Cost (Bil $)','series':'cost_cat_bulk','chart_type':'Bar', 'filter': {'cost_cat_bulk':{'exclude':costs_pol_inv}}}),
            ('2018-end Discounted',{'x':'scenario','y':'Discounted Cost (Bil $)','series':'cost_cat_bulk','chart_type':'Bar', 'filter': {'cost_cat_bulk':{'exclude':costs_orig_inv}, 'year': {'start':2018}}}),
            ('Discounted by Year',{'x':'year','y':'Discounted Cost (Bil $)','series':'cost_cat_bulk','explode':'scenario','chart_type':'Bar', 'bar_width':'1.75', 'filter': {'cost_cat_bulk':{'exclude':costs_orig_inv}}}),
            ('Undiscounted by Year',{'x':'year','y':'Cost (Bil $)','series':'cost_cat_bulk','explode':'scenario','chart_type':'Bar', 'bar_width':'1.75', 'filter': {'cost_cat_bulk':{'exclude':costs_orig_inv}}}),
        )),
        }
    ),

    ('Value Streams chosen',
        {'file': 'valuestreams_chosen.csv',
        'columns': ['tech', 'vintage', 'n', 'year','new_old', 'var_name', 'con_name', 'value'],
        'preprocess': [
            {'func': pre_val_streams, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('$ by type over time', {'x':'year','y':'value','series':'con_name', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'new_old':['new']}}),
            ('$ by type final', {'chart_type':'Bar', 'x':'tech', 'y':'value', 'series':'con_name', 'explode':'scenario', 'sync_axes':'No', 'bar_width':r'.9', 'cum_sort': 'Descending', 'plot_width':'600', 'plot_height':'600', 'filter': {'new_old':['new'], 'con_name':{'exclude':['mwh','kw']}, 'year':'last', }}),

            ('$/kW by type over time', {'x':'year','y':'value','series':'con_name', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'con_name', 'adv_col_base':'kw', 'chart_type':'Bar', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'con_name':{'exclude':['mwh']},'new_old':['new']}}),
            ('$/kW by type final', {'chart_type':'Bar', 'x':'tech', 'y':'value', 'series':'con_name', 'explode':'scenario', 'adv_op':'Ratio', 'adv_col':'con_name', 'adv_col_base':'kw', 'sync_axes':'No', 'bar_width':r'.9', 'cum_sort': 'Descending', 'plot_width':'600', 'plot_height':'600', 'filter': {'new_old':['new'], 'con_name':{'exclude':['mwh']}, 'year':'last', }}),

            ('$/MWh by type over time', {'x':'year','y':'value','series':'con_name', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'con_name', 'adv_col_base':'mwh', 'chart_type':'Bar', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'con_name':{'exclude':['kw']},'new_old':['new']}}),
            ('$/MWh by type final', {'chart_type':'Bar', 'x':'tech', 'y':'value', 'series':'con_name', 'explode':'scenario', 'adv_op':'Ratio', 'adv_col':'con_name', 'adv_col_base':'mwh', 'sync_axes':'No', 'bar_width':r'.9', 'cum_sort': 'Descending', 'plot_width':'600', 'plot_height':'600', 'filter': {'new_old':['new'], 'con_name':{'exclude':['kw']}, 'year':'last', }}),
        )),
        }
    ),

    ('Reduced Cost ($/kW)',
        {'file': 'reduced_cost.csv',
        'columns': ['tech', 'vintage', 'region', 'year','bin','$/kW'],
        'preprocess': [
            {'func': pre_reduced_cost, 'args': {}},
            {'func': map_i_to_n, 'args': {}},
            {'func': apply_inflation, 'args': {'column': '$/kW'}},
        ],
        'presets': collections.OrderedDict((
            ('Final supply curves', {'chart_type':'Dot', 'x':'icrb', 'y':'$/kW', 'explode':'scenario','explode_group':'tech', 'sync_axes':'No', 'cum_sort': 'Ascending', 'plot_width':'600', 'plot_height':'600', 'filter': {'year':'last', }}),
            ('Final supply curves p1', {'chart_type':'Dot', 'x':'icrb', 'y':'$/kW', 'explode':'scenario','explode_group':'tech', 'sync_axes':'No', 'cum_sort': 'Ascending', 'plot_width':'600', 'plot_height':'600', 'filter': {'year':'last', 'n':['p1']}}),
        )),
        }
    ),

    ('LCOE ($/MWh)',
        {'sources': [
            {'name': 'lcoe', 'file': 'lcoe.csv', 'columns': ['tech', 'vintage', 'region', 'year', 'bin','$/MWh']},
            {'name': 'inv', 'file': 'cap_new_bin_out.csv', 'columns': ['tech', 'vintage', 'region', 'year', 'bin','chosen MW']},
            {'name': 'avail', 'file': 'cap_avail.csv', 'columns': ['tech', 'region', 'year', 'bin','available MW']},
        ],
        'preprocess': [
            {'func': pre_lcoe, 'args': {}},
            {'func': map_i_to_n, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Final supply curves', {'chart_type':'Dot', 'x':'icrb', 'y':'$/MWh', 'explode':'scenario','explode_group':'tech', 'sync_axes':'No', 'cum_sort': 'Ascending', 'plot_width':'600', 'plot_height':'600', 'filter': {'year':'last', }}),
            ('Final supply curves p1', {'chart_type':'Dot', 'x':'icrb', 'y':'$/MWh', 'explode':'scenario','explode_group':'tech', 'sync_axes':'No', 'cum_sort': 'Ascending', 'plot_width':'600', 'plot_height':'600', 'filter': {'year':'last', 'n':['p1']}}),
            ('Final supply curves chosen', {'chart_type':'Dot', 'x':'icrb', 'y':'$/MWh', 'explode':'scenario','explode_group':'tech', 'sync_axes':'No', 'cum_sort': 'Ascending', 'plot_width':'600', 'plot_height':'600', 'filter': {'year':'last', 'chosen':['yes']}}),
            ('Final supply curves chosen p1', {'chart_type':'Dot', 'x':'icrb', 'y':'$/MWh', 'explode':'scenario','explode_group':'tech', 'sync_axes':'No', 'cum_sort': 'Ascending', 'plot_width':'600', 'plot_height':'600', 'filter': {'year':'last', 'chosen':['yes'], 'n':['p1']}}),
        )),
        }
    ),

    ('LCOE nopol ($/MWh)',
        {'sources': [
            {'name': 'lcoe', 'file': 'lcoe_nopol.csv', 'columns': ['tech', 'vintage', 'region', 'year', 'bin','$/MWh']},
            {'name': 'inv', 'file': 'cap_new_bin_out.csv', 'columns': ['tech', 'vintage', 'region', 'year', 'bin','chosen MW']},
            {'name': 'avail', 'file': 'cap_avail.csv', 'columns': ['tech', 'region', 'year', 'bin','available MW']},
        ],
        'preprocess': [
            {'func': pre_lcoe, 'args': {}},
            {'func': map_i_to_n, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Final supply curves', {'chart_type':'Dot', 'x':'icrb', 'y':'$/MWh', 'explode':'scenario','explode_group':'tech', 'sync_axes':'No', 'cum_sort': 'Ascending', 'plot_width':'600', 'plot_height':'600', 'filter': {'year':'last', }}),
            ('Final supply curves p1', {'chart_type':'Dot', 'x':'icrb', 'y':'$/MWh', 'explode':'scenario','explode_group':'tech', 'sync_axes':'No', 'cum_sort': 'Ascending', 'plot_width':'600', 'plot_height':'600', 'filter': {'year':'last', 'n':['p1']}}),
            ('Final supply curves chosen', {'chart_type':'Dot', 'x':'icrb', 'y':'$/MWh', 'explode':'scenario','explode_group':'tech', 'sync_axes':'No', 'cum_sort': 'Ascending', 'plot_width':'600', 'plot_height':'600', 'filter': {'year':'last', 'chosen':['yes']}}),
            ('Final supply curves chosen p1', {'chart_type':'Dot', 'x':'icrb', 'y':'$/MWh', 'explode':'scenario','explode_group':'tech', 'sync_axes':'No', 'cum_sort': 'Ascending', 'plot_width':'600', 'plot_height':'600', 'filter': {'year':'last', 'chosen':['yes'], 'n':['p1']}}),
        )),
        }
    ),

    ('LCOE fullpol ($/MWh)',
        {'sources': [
            {'name': 'lcoe', 'file': 'lcoe_fullpol.csv', 'columns': ['tech', 'vintage', 'region', 'year', 'bin','$/MWh']},
            {'name': 'inv', 'file': 'cap_new_bin_out.csv', 'columns': ['tech', 'vintage', 'region', 'year', 'bin','chosen MW']},
            {'name': 'avail', 'file': 'cap_avail.csv', 'columns': ['tech', 'region', 'year', 'bin','available MW']},
        ],
        'preprocess': [
            {'func': pre_lcoe, 'args': {}},
            {'func': map_i_to_n, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Final supply curves', {'chart_type':'Dot', 'x':'icrb', 'y':'$/MWh', 'explode':'scenario','explode_group':'tech', 'sync_axes':'No', 'cum_sort': 'Ascending', 'plot_width':'600', 'plot_height':'600', 'filter': {'year':'last', }}),
            ('Final supply curves p1', {'chart_type':'Dot', 'x':'icrb', 'y':'$/MWh', 'explode':'scenario','explode_group':'tech', 'sync_axes':'No', 'cum_sort': 'Ascending', 'plot_width':'600', 'plot_height':'600', 'filter': {'year':'last', 'n':['p1']}}),
            ('Final supply curves chosen', {'chart_type':'Dot', 'x':'icrb', 'y':'$/MWh', 'explode':'scenario','explode_group':'tech', 'sync_axes':'No', 'cum_sort': 'Ascending', 'plot_width':'600', 'plot_height':'600', 'filter': {'year':'last', 'chosen':['yes']}}),
            ('Final supply curves chosen p1', {'chart_type':'Dot', 'x':'icrb', 'y':'$/MWh', 'explode':'scenario','explode_group':'tech', 'sync_axes':'No', 'cum_sort': 'Ascending', 'plot_width':'600', 'plot_height':'600', 'filter': {'year':'last', 'chosen':['yes'], 'n':['p1']}}),
        )),
        }
    ),
))
