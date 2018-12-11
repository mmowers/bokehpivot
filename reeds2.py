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

rb_globs = {'output_subdir': '\\outputs\\', 'test_file': 'cap.csv'}
this_dir_path = os.path.dirname(os.path.realpath(__file__))
CRF_reeds = 0.077
df_deflator = pd.read_csv(this_dir_path + '/in/inflation.csv', index_col=0)
ILR_UPV = 1.3
ILR_distPV = 1.1

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

def strip_s_from_region(df, **kw):
    df['region'] = df['region'].map(lambda x: x.lstrip('s'))
    return df

def map_i_to_n(df, **kw):
    df_hier = pd.read_csv(this_dir_path + '/in/hierarchy.csv')
    dict_hier = dict(zip(df_hier['i'].astype(str), df_hier['n']))
    df['region'] = df['region'].replace(dict_hier)
    df.rename(columns={'region': 'n'}, inplace=True)
    return df


def remove_n(df, **kw):
    df = df[~df['region'].astype(str).str.startswith('p')].copy()
    df.rename(columns={'region': 'i'}, inplace=True)
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
            {'func': strip_s_from_region, 'args': {}},
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
            {'func': strip_s_from_region, 'args': {}},
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
            {'func': strip_s_from_region, 'args': {}},
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
            {'func': strip_s_from_region, 'args': {}},
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
            {'func': strip_s_from_region, 'args': {}},
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
        	{'func': strip_s_from_region, 'args': {}},
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
            ('2017-end Stacked Bars',{'x':'scenario','y':'Discounted Cost (Bil $)','series':'cost_cat','chart_type':'Bar', 'filter': {'year': {'start':2017}}}),
        )),
        }
    ),

    ('Value Streams chosen raw',
        {'file': 'valuestreams_chosen.csv',
        'columns': ['year', 'tech', 'new_old', 'region', 'type', 'timeslice', '$'],
        'preprocess': [
            {'func': strip_s_from_region, 'args': {}},
            {'func': map_i_to_n, 'args': {}},
            {'func': sum_over_cols, 'args': {'group_cols': ['year', 'tech', 'new_old', 'n', 'type'], 'sum_over_cols': ['timeslice']}},
            {'func': apply_inflation, 'args': {'column': '$'}},
            {'func': scale_column, 'args': {'scale_factor': 1e-9, 'column': '$'}},
        ],
        'presets': collections.OrderedDict((
            ('Bil $ by type over time', {'x':'year','y':'$','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'sync_axes':'No'}),
            ('New Bil $ by type over time', {'x':'year','y':'$','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'new_old':['new']}}),
            ('Old Bil $ by type over time', {'x':'year','y':'$','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'new_old':['old']}}),
            ('Mixed Bil $ by type over time', {'x':'year','y':'$','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'new_old':['mixed']}}),
            ('Retire Bil $ by type over time', {'x':'year','y':'$','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'new_old':['retire']}}),
            ('New Bil $ Cost by tech over time', {'x':'year','y':'$','series':'tech', 'explode': 'scenario', 'chart_type':'Bar', 'bar_width':'1.75', 'y_scale':'-1', 'filter': {'new_old':['new'], 'type':['fix_cost','gp','trans_cost','var_cost']}}),
            ('New Bil $ by type over time agg', {'x':'year','y':'$','series':'type', 'explode': 'scenario', 'chart_type':'Bar', 'bar_width':'1.75', 'filter': {'new_old':['new']}}),
        )),
        }
    ),

))
