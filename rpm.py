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
import rpm_valuestreams as rpmvs

rb_globs = {'output_subdir': '\\inout\\', 'test_file': 'Dispatch_allyrs.gdx', 'report_subdir':'/rpm'}
this_dir_path = os.path.dirname(os.path.realpath(__file__))

CRF_reeds = 0.077
df_deflator = pd.read_csv(this_dir_path + '/in/inflation.csv', index_col=0)
#ILR_UPV = 1.3
#ILR_distPV = 1.1

raw_costs = ['fix_cost','var_cost','trans_cost','gp','oper_res_cost','other_cost']
costs = ['Fixed Cost','Variable Cost','Trans Cost','Growth Cost','Ancillary Cost','Other Cost']
raw_values = ['load_pca','res_marg','oper_res','rps','cap_fo_po','surplus','other']
values = ['Energy Value','Capacity Value','Ancillary Value','RPS Value','Cap Fo Po','Curtailment','Other Value']

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

#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------

#2. Columns metadata. These are columns that are referenced in the Results section below.
#This is where joins, maps, and styles are applied for the columns.
#For 'style', colors are in hex, but descriptions are given (see http://www.color-hex.com/color-names.html).
columns_meta = {
    'tech':{
        'type': 'string',
        'map': this_dir_path + '/in/rpm/tech_map.csv',
        'style': this_dir_path + '/in/rpm/tech_style.csv',
    },
    'class':{
        'type': 'string',
    },
    'region':{
        'type': 'string',
    },
    'year':{
        'type': 'number',
        'filterable': True,
        'seriesable': True,
        'y-allow': False,
    },
    'cost_val_type':{
        'type': 'string',
        'map': this_dir_path + '/in/rpm/cost_val_map.csv',
        'style': this_dir_path + '/in/rpm/cost_val_style.csv',
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
        {'file': 'Dispatch_allyrs.gdx',
        'param': 'capacity_ba_allyrs',
        'columns': ['type','region','tech', 'class', 'year', 'Capacity (GW)'],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': .001, 'column': 'Capacity (GW)'}},
            {'func': sum_over_cols, 'args': {'sum_over_cols': ['class'], 'group_cols': ['type','region','tech', 'year']}},
        ],
        'index': ['type', 'region', 'tech', 'year'],
        'presets': collections.OrderedDict((
            ('Stacked Area Generation Capacity (GW)',{'x':'year','y':'Capacity (GW)','series':'tech', 'explode': 'scenario','chart_type':'Area','filter': {'type': ['generation']}}),
            ('Stacked Bars Generation Capacity (GW)',{'x':'year','y':'Capacity (GW)','series':'tech', 'explode': 'scenario','chart_type':'Bar', 'bar_width':'4.5','filter': {'type': ['generation']}}),
            ('Stacked Bars Generation Capacity Explode by BA (GW)',{'x':'year','y':'Capacity (GW)','series':'tech', 'explode': 'scenario', 'explode_group': 'region', 'sync_axes': 'No', 'chart_type':'Bar', 'bar_width':'4.5','filter': {'type': ['generation']}}),
        )),
        }
    ),

	('Value Streams BA',
        {'file': 'valuestreams/valuestreams_chosen_ba.csv',
		'columns': ['year','tech','class','new_old','ba','cost_val_type','$'],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': 1/1e9, 'column': '$'}},
        ],
        'presets': collections.OrderedDict((
		    ('WECC by Tech, Class over Time', {'x':'year', 'x_group':'class', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'explode_group':'tech', 'chart_type':'Bar', 'bar_width':'0.75', 'sync_axes':'No'}),
			('LDWP by Tech, Class over Time', {'x':'year', 'x_group':'class', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'explode_group':'tech', 'chart_type':'Bar', 'bar_width':'0.75', 'sync_axes':'No', 'filter':{'ba':['ldwp']}}),

            ('Geo by Class, BA over Time', {'x':'year', 'x_group':'ba', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'explode_group':'class', 'chart_type':'Bar', 'bar_width':'0.75', 'sync_axes':'No', 'filter':{'tech':['geo']}}),
            ('Geo by Class over Time', {'x':'year', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'explode_group':'class', 'chart_type':'Bar', 'bar_width':'0.75', 'sync_axes':'No', 'filter':{'tech':['geo']}}),

            ('Wind by Class, BA over Time', {'x':'year', 'x_group':'ba', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'explode_group':'class', 'chart_type':'Bar', 'bar_width':'0.75', 'sync_axes':'No', 'filter':{'tech':['wind']}}),
            ('Wind by Class over Time', {'x':'year', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'explode_group':'class', 'chart_type':'Bar', 'bar_width':'0.75', 'sync_axes':'No', 'filter':{'tech':['wind']}}),

			('By Tech over Time', {'x':'year', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'explode_group':'tech', 'chart_type':'Bar', 'bar_width':'0.75', 'sync_axes':'No'}),

            ('By BA over Time', {'x':'year', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'explode_group':'ba', 'chart_type':'Bar', 'bar_width':'0.75', 'sync_axes':'No'}),
        )),
        }
    ),

	('Value Streams Node',
        {'file': 'valuestreams/valuestreams_chosen_node.csv',
	    'columns': ['year','tech','class','new_old','node','cost_val_type','$'],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': 1/1e9, 'column': '$'}},
        ],
        'presets': collections.OrderedDict((
            ('LDWP by Node over Time', {'x':'node', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'explode_group':'year', 'chart_type':'Bar', 'bar_width':'0.75', 'sync_axes':'No'}),
        )),
        }
    ),
))
