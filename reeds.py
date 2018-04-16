'''
ReEDS results metadata and preprocess functions.

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

this_dir_path = os.path.dirname(os.path.realpath(__file__))
inflation_mult = 1.2547221 #2004$ to 2015$
CRF_reeds = 0.0832902994316595

#1. Preprocess functions for results_meta
def scale_column(df, **kw):
    df[kw['column']] = df[kw['column']] * kw['scale_factor']
    return df

def scale_column_filtered(df, **kw):
    cond = df[kw['by_column']].isin(kw['by_vals'])
    df.loc[cond, kw['change_column']] = df.loc[cond, kw['change_column']] * kw['scale_factor']
    return df

def sum_over_cols(df, **kw):
    df.drop(kw['sum_over_cols'], axis='columns', inplace=True)
    df =  df.groupby(kw['group_cols'], sort=False, as_index =False).sum()
    return df

def discount_costs(df, **kw):
    #inner join the cost_cat_type.csv table to get types of costs (Capital, Operation)
    cost_cat_type = pd.read_csv(this_dir_path + '/in/cost_cat_type.csv')
    df = pd.merge(left=df, right=cost_cat_type, on='cost_cat', sort=False)
    #make new column that is the pv multiplier
    df['pv_mult'] = df.apply(lambda x: get_pv_mult(int(x['year']), x['type']), axis=1)
    df['Discounted Cost (Bil 2015$)'] = df['Cost (Bil 2015$)'] * df['pv_mult']
    return df

#Return present value multiplier
def get_pv_mult(year, type, dinvest=0.054439024, dsocial=0.03, lifetime=20, refyear=2017, lastyear=2100):
    if type == "Operation":
        pv_mult = 1 / (1 + dsocial)**(year - refyear)
    elif type == "Capital":
        pv_mult = CRF(dinvest, lifetime) / CRF(dinvest, min(lifetime, lastyear + 1 - year)) * 1 / (1 + dsocial)**(year - refyear)
    return pv_mult

#Capital recovery factor
def CRF(i,n):
    return i/(1-(1/(1+i)**n))

def pre_elec_price(df, **kw):
    df = df.pivot_table(index=['n','year'], columns='elem', values='value').reset_index()
    df.drop(['t2','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16'], axis='columns', inplace=True)
    df.columns.name = None
    df.rename(columns={'t1': 'load', 't3': 'Regulated', 't17': 'Competitive'}, inplace=True)
    df = pd.melt(df, id_vars=['n', 'year', 'load'], value_vars=['Competitive', 'Regulated'], var_name='type', value_name= 'Price (2015$/MWh)')
    df['Price (2015$/MWh)'] = df['Price (2015$/MWh)'] * inflation_mult
    return df

def pre_elec_price_components(dfs, **kw):
    df_load = dfs['load'][dfs['load']['type'] == 'reqt']
    df_load = df_load.drop('type', 1)
    df_main = dfs['main']
    df_main['value'] = df_main['value'] * inflation_mult
    df = pd.merge(left=df_main, right=df_load, how='inner', on=['n','year'], sort=False)
    return df

def pre_marginal_capacity_value(dfs, **kw):
    dfs['new_cap']['Capacity (GW)'] = dfs['new_cap']['Capacity (GW)'] / 1000
    df = pd.merge(left=dfs['cv_mar'], right=dfs['new_cap'], on=['tech','n','year'], how='left', sort=False)
    return df

def pre_marginal_curtailment_filter(df, **kw):
    df = df[df['type'].isin(['surpmar','new-generation'])]
    return df

def pre_marginal_curtailment(df, **kw):
    index_cols = [i for i in df.columns.values.tolist() if i not in ['type','value']]
    df = df.pivot_table(index=index_cols, columns='type', values='value').reset_index()
    df.rename(columns={'surpmar': 'surplus', 'new-generation': 'gen'}, inplace=True)
    df['surpmar'] = df['surplus'] / df['gen']
    return df

def pre_tech_val_streams_potential(dfs, **kw):
    #Add a MWh/kW row for each resource for $/MWh calc, and add block revenue in $/kW for value factor calc.
    #All $ are annualized
    df_valstream = dfs['valstream']
    df_load = dfs['load']
    df_price = dfs['prices_nat']
    #convert to annualized $/kW
    df_valstream['$/kW'] = df_valstream['$/kW'] * inflation_mult * CRF_reeds
    df_load['type'] = 'MWh/kW'
    df_price = df_price[df_price['type'].isin(['load_pca','res_marg'])].copy()
    #sum load and res_marg prices
    df_price = sum_over_cols(df_price, sum_over_cols=['type'], group_cols=['year'])
    #merge df_load into df_price
    df_price['year'] = pd.to_numeric(df_price['year'])
    df_price = pd.merge(left=df_price, right=df_load, on=['year'], how='left', sort=False)
    df_price['$/kW'] = df_price['$/MWh'] * df_price['MWh/kW'] * inflation_mult
    df_price['type'] = 'block_revenue'
    df_price.drop(['$/MWh','MWh/kW'], axis='columns', inplace=True)
    df_load.rename(columns={'MWh/kW': '$/kW'}, inplace=True) #rename just so we can concatenate, even though units are MWh/kW
    df = pd.concat([df_valstream,df_load,df_price], ignore_index=True)
    return df

def pre_stacked_profitability_potential(df, **kw):
    #Sum all costs so that we can calculate value / total cost for each value stream
    #remove quantity
    #label all costs the same so they can be grouped
    costs = ['fix_cost','var_cost','trans_cost','gp']
    df.loc[df['type'].isin(costs),'type'] = 'cost'
    df.loc[df['type'] == 'cost','$/kW'] *= -1
    #sum costs
    df =  df.groupby(['tech', 'new_old', 'year', 'n','type','var_set'], sort=False, as_index =False).sum()
    return df

def add_huc_reg(df, **kw):
    huc_map = pd.read_csv(this_dir_path + '/in/huc_2_ratios.csv', dtype={'huc_2':object})
    df = pd.merge(left=df, right=huc_map, how='outer', on='n', sort=False)
    df['value'] = df['value'] * df['pca_huc_ratio']
    df = df.drop('pca_huc_ratio', 1)
    df = df.drop('huc_pca_ratio', 1)
    return df

#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------

#2. Columns metadata. These are columns that are referenced in the Results section below.
#This is where joins, maps, and styles are applied for the columns.
#For 'style', colors are in hex, but descriptions are given (see http://www.color-hex.com/color-names.html).
columns_meta = {
    'tech':{
        'type': 'string',
        'map': this_dir_path + '/in/tech_map.csv',
        'style': this_dir_path + '/in/tech_style.csv',
    },
    'jedi_tech':{
        'type': 'string',
        'style': this_dir_path + '/in/jedi_tech_style.csv',
    },
    'directness':{
        'type': 'string',
        'map': this_dir_path + '/in/jedi_directness_map.csv',
        'style': this_dir_path + '/in/jedi_directness_style.csv',
    },
    'i':{
        'type': 'string',
        'join': this_dir_path + '/in/hierarchy.csv',
    },
    'n':{
        'type': 'string',
        'join': this_dir_path + '/in/hierarchy.csv',
    },
    'state_plus_dc':{
        'type': 'string',
        'join': this_dir_path + '/in/hierarchy_st_plus_dc.csv',
    },
    'huc_2':{
        'type': 'string',
        'join': this_dir_path + '/in/huc_join.csv',
    },
    'huc_4':{
        'type': 'string',
        'join': this_dir_path + '/in/huc_join.csv',
    },
    'huc_6':{
        'type': 'string',
        'join': this_dir_path + '/in/huc_join.csv',
    },
    'huc_8':{
        'type': 'string',
        'join': this_dir_path + '/in/huc_join.csv',
    },
    'year':{
        'type': 'number',
        'filterable': True,
        'seriesable': True,
        'y-allow': False,
    },
    'm':{
        'type': 'string',
        'style': this_dir_path + '/in/m_style.csv',
    },
    'cost_cat':{
        'type': 'string',
        'map': this_dir_path + '/in/cost_cat_map.csv',
        'style': this_dir_path + '/in/cost_cat_style.csv',
    },
    'tech_val_type':{
        'map': this_dir_path + '/in/tech_val_type_map.csv',
        'style': this_dir_path + '/in/tech_val_type_style.csv',
    },
    'val_stream_type':{
        'map': this_dir_path + '/in/val_stream_type_map.csv',
        'style': this_dir_path + '/in/val_stream_type_style.csv',
    },
}

#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------

#3. Results metadata. This is where all ReEDS results are defined. Parameters are read from gdx files, and
#are converted into pandas dataframes for pivoting. Preprocess functions may be used to perform additional manipulation.
#Note that multiple parameters may be read in for the same result (search below for 'sources')
#Presets may also be defined.
results_meta = collections.OrderedDict((
    ('Capacity (GW)',
        {'file': 'CONVqn.gdx',
        'param': 'CONVqnallyears',
        'columns': ['tech', 'n', 'year', 'Capacity (GW)'],
        'index': ['tech','n','year'],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': .001, 'column': 'Capacity (GW)'}},
            {'func': scale_column_filtered, 'args': {'by_column': 'tech', 'by_vals': ['upv', 'dupv', 'distpv'], 'change_column': 'Capacity (GW)', 'scale_factor': 1/1.1}},
        ],
        'presets': collections.OrderedDict((
            ('Stacked Area',{'x':'year','y':'Capacity (GW)','series':'tech', 'explode': 'scenario','chart_type':'Area'}),
            ('Stacked Bars',{'x':'year','y':'Capacity (GW)','series':'tech', 'explode': 'scenario','chart_type':'Bar', 'bar_width':'1.75'}),
            ('Explode By Tech',{'x':'year','y':'Capacity (GW)','series':'scenario', 'explode': 'tech','chart_type':'Line'}),
            ('Stacked Col Sel years',{'x':'scenario','y':'Capacity (GW)','series':'tech', 'explode': 'year', 'chart_type':'Bar', 'filter': {'year': [2020, 2030, 2050]}}),
            ('State Map 2030 Wind',{'x':'st','y':'Capacity (GW)', 'explode': 'scenario','chart_type':'Map', 'filter': {'tech': ['Wind'], 'year': [2030]}}),
            ('State Map 2030 Solar',{'x':'st','y':'Capacity (GW)', 'explode': 'scenario','chart_type':'Map', 'filter': {'tech': ['PV (AC)', 'Distributed PV (AC)', 'CSP'], 'year': [2030]}}),
            ('State Map Final Wind',{'x':'st','y':'Capacity (GW)', 'explode': 'scenario','chart_type':'Map', 'filter': {'tech': ['Wind'], 'year': 'last'}}),
            ('State Map Final Solar',{'x':'st','y':'Capacity (GW)', 'explode': 'scenario','chart_type':'Map', 'filter': {'tech': ['PV (AC)', 'Distributed PV (AC)', 'CSP'], 'year': 'last'}}),
            ('PCA Map 2030 Wind',{'x':'n','y':'Capacity (GW)', 'explode': 'scenario','chart_type':'Map', 'filter': {'tech': ['Wind'], 'year': [2030]}}),
            ('PCA Map 2030 Solar',{'x':'n','y':'Capacity (GW)', 'explode': 'scenario','chart_type':'Map', 'filter': {'tech': ['PV (AC)', 'Distributed PV (AC)', 'CSP'], 'year': [2030]}}),
            ('PCA Map Final Wind',{'x':'n','y':'Capacity (GW)', 'explode': 'scenario','chart_type':'Map', 'filter': {'tech': ['Wind'], 'year': 'last'}}),
            ('PCA Map Final Solar',{'x':'n','y':'Capacity (GW)', 'explode': 'scenario','chart_type':'Map', 'filter': {'tech': ['PV (AC)', 'Distributed PV (AC)', 'CSP'], 'year': 'last'}}),
        )),
        }
    ),
    ('New Capacity (GW)',
        {'file': 'CONVqn.gdx',
        'param': 'CONVqn_newallyears',
        'columns': ['tech', 'n', 'year', 'Capacity (GW)'],
        'index': ['tech','n','year'],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': .001, 'column': 'Capacity (GW)'}},
            {'func': scale_column_filtered, 'args': {'by_column': 'tech', 'by_vals': ['upv', 'dupv', 'distpv'], 'change_column': 'Capacity (GW)', 'scale_factor': 1/1.1}},
        ],
        'presets': collections.OrderedDict((
            ('Stacked Bars',{'x':'year','y':'Capacity (GW)','series':'tech', 'explode': 'scenario','chart_type':'Bar', 'bar_width':'1.75'}),
            ('Explode By Tech',{'x':'year','y':'Capacity (GW)','series':'scenario', 'explode': 'tech','chart_type':'Line'}),
        )),
        }
    ),
    ('Retirements (GW)',
        {'file': "CONVqn.gdx",
        'param': 'Retireqnallyears',
        'columns': ["tech", "n", "year", "Capacity (GW)"],
        'index': ['tech','n','year'],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': .001, 'column': 'Capacity (GW)'}},
            {'func': scale_column_filtered, 'args': {'by_column': 'tech', 'by_vals': ['upv', 'dupv', 'distpv'], 'change_column': 'Capacity (GW)', 'scale_factor': 1/1.1}},
        ],
        'presets': collections.OrderedDict((
            ('Stacked Bars',{'x':'year','y':'Capacity (GW)','series':'tech', 'explode': 'scenario','chart_type':'Bar', 'bar_width':'1.5'}),
            ('Explode By Tech',{'x':'year','y':'Capacity (GW)','series':'scenario', 'explode': 'tech','chart_type':'Line'}),
        )),
        }
    ),
    ('Wind Capacity (GW)',
        {'file': 'CONVqn.gdx',
        'param': 'Windiallc',
        'columns': ["windtype", "i", "year", "class", "Capacity (GW)"],
        'index': ["windtype", "i", "year", "class"],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': .001, 'column': 'Capacity (GW)'}},
        ],
        'presets': collections.OrderedDict((
            ('Final Map',{'x':'i','y':'Capacity (GW)', 'explode': 'scenario','chart_type':'Map', 'filter': {'year': 'last'}}),
        )),
        }
    ),
    ('Generation (TWh)',
        {'file': 'CONVqn.gdx',
        'param': 'CONVqmnallyears',
        'columns': ['tech', 'n', 'year', 'Generation (TWh)'],
        'index': ['tech','n','year'],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': 0.000001, 'column': 'Generation (TWh)'}},
        ],
        'presets': collections.OrderedDict((
            ('Stacked Area',{'x':'year','y':'Generation (TWh)','series':'tech', 'explode': 'scenario','chart_type':'Area'}),
            ('Stacked Bars',{'x':'year','y':'Generation (TWh)','series':'tech', 'explode': 'scenario','chart_type':'Bar', 'bar_width':'1.75'}),
            ('Stacked Bars Gen Frac',{'x':'year','y':'Generation (TWh)','series':'tech', 'explode': 'scenario','adv_op':'Ratio','adv_col':'tech','adv_col_base':'Total','chart_type':'Bar', 'bar_width':'1.75'}),
            ('Explode By Tech',{'x':'year','y':'Generation (TWh)','series':'scenario', 'explode': 'tech','chart_type':'Line'}),
            ('Stacked Columns 2020, 2030, 2050',{'x':'scenario','y':'Generation (TWh)','series':'tech', 'explode': 'year', 'chart_type':'Bar', 'filter': {'year': [2020, 2030, 2050]}}),
            ('State Map 2030 Wind',{'x':'st','y':'Generation (TWh)', 'explode': 'scenario','chart_type':'Map', 'filter': {'tech': ['Wind'], 'year': [2030]}}),
            ('State Map 2030 Solar',{'x':'st','y':'Generation (TWh)', 'explode': 'scenario','chart_type':'Map', 'filter': {'tech': ['PV (AC)', 'Distributed PV (AC)', 'CSP'], 'year': [2030]}}),
            ('State Map Final Wind',{'x':'st','y':'Generation (TWh)', 'explode': 'scenario','chart_type':'Map', 'filter': {'tech': ['Wind'], 'year': 'last'}}),
            ('State Map Final Solar',{'x':'st','y':'Generation (TWh)', 'explode': 'scenario','chart_type':'Map', 'filter': {'tech': ['PV (AC)', 'Distributed PV (AC)', 'CSP'], 'year': 'last'}}),
            ('PCA Map 2030 Wind',{'x':'n','y':'Generation (TWh)', 'explode': 'scenario','chart_type':'Map', 'filter': {'tech': ['Wind'], 'year': [2030]}}),
            ('PCA Map 2030 Solar',{'x':'n','y':'Generation (TWh)', 'explode': 'scenario','chart_type':'Map', 'filter': {'tech': ['PV (AC)', 'Distributed PV (AC)', 'CSP'], 'year': [2030]}}),
            ('PCA Map Final Wind',{'x':'n','y':'Generation (TWh)', 'explode': 'scenario','chart_type':'Map', 'filter': {'tech': ['Wind'], 'year': 'last'}}),
            ('PCA Map Final Solar',{'x':'n','y':'Generation (TWh)', 'explode': 'scenario','chart_type':'Map', 'filter': {'tech': ['PV (AC)', 'Distributed PV (AC)', 'CSP'], 'year': 'last'}}),
        )),
        }
    ),
    ('Emissions, Fuel, Prices',
        {'file': 'Reporting.gdx',
        'param': 'AnnualReport',
        'columns': ['n', 'year', 'type', 'value'],
        'index': ['n','year','type'],
        'presets': collections.OrderedDict((
            ('CO2 Emissions (MMton)',{'x':'year','y':'value','series':'scenario','chart_type':'Line', 'filter': {'type': ['co2']}, 'y_scale':'1e-6'}),
        )),
        }
    ),
    ('Sys Cost (Bil 2015$)',
        {'file': 'systemcost.gdx',
        'param': 'aSystemCost_ba',
        'columns': ['cost_cat', 'n', 'year', 'Cost (Bil 2015$)'],
        'index': ['cost_cat', 'n', 'year'],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': inflation_mult/1e9, 'column': 'Cost (Bil 2015$)'}},
            {'func': discount_costs, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Stacked Bars',{'x':'scenario','y':'Discounted Cost (Bil 2015$)','series':'cost_cat','chart_type':'Bar'}),
            ('2017-end Stacked Bars',{'x':'scenario','y':'Discounted Cost (Bil 2015$)','series':'cost_cat','chart_type':'Bar', 'filter': {'year': {'start':2017}}}),
        )),
        }
    ),
    ('Gen by m (GW)',
        {'file': 'CONVqn.gdx',
        'param': 'CONVqmnallm',
        'columns': ['tech', 'n', 'year', 'm', 'Generation (GW)'],
        'index': ['tech', 'year', 'm'],
        'preprocess': [
            {'func': sum_over_cols, 'args': {'group_cols': ['tech', 'year', 'm'], 'sum_over_cols': ['n']}},
            {'func': scale_column, 'args': {'scale_factor': .001, 'column': 'Generation (GW)'}},
        ],
        'presets': collections.OrderedDict((
            ('Stacked Bars Final',{'x':'m','y':'Generation (GW)','series':'tech', 'explode': 'scenario','chart_type':'Bar', 'filter': {'year': 'last'}}),
        )),
        }
    ),
    ('Gen by m full (GW)',
        {'file': 'CONVqn.gdx',
        'param': 'CONVqmnallm',
        'columns': ['tech', 'n', 'year', 'm', 'Generation (GW)'],
        'index': ['tech', 'n', 'year', 'm'],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': .001, 'column': 'Generation (GW)'}},
        ],
        'presets': collections.OrderedDict((
            ('Stacked Bars Final',{'x':'m','y':'Generation (GW)','series':'tech', 'explode': 'scenario','chart_type':'Bar', 'filter': {'year': 'last'}}),
        )),
        }
    ),
    ('Elec Price (2015$/MWh)',
        {'file': 'Reporting.gdx',
        'param': 'ElecPriceOut',
        'columns': ['n', 'year', 'elem', 'value'],
        'index': ['n', 'year', 'type'],
        'preprocess': [
            {'func': pre_elec_price, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('National',{'x':'year','y':'Price (2015$/MWh)', 'y_agg':'Weighted Ave', 'y_weight':'load', 'series':'scenario', 'explode': 'type', 'chart_type':'Line'}),
            ('National Scenario',{'x':'year','y':'Price (2015$/MWh)', 'y_agg':'Weighted Ave', 'y_weight':'load', 'series':'type', 'explode': 'scenario', 'chart_type':'Line'}),
            ('Census',{'x':'year','y':'Price (2015$/MWh)', 'y_agg':'Weighted Ave', 'y_weight':'load', 'series':'scenario', 'explode': 'censusregions', 'explode_group': 'type', 'chart_type':'Line'}),
        )),
        }
    ),
    ('Elec Price Components',
        {'sources': [
            {'name': 'load', 'file': 'CONVqn.gdx', 'param': 'CONVqmnallyears', 'columns': ['type', 'n', 'year', 'load']},
            {'name': 'main', 'file': 'MarginalPrices.gdx', 'param': 'wpmarg_BA_ann_allyrs', 'columns': ['n', 'elec_comp_type','year', 'value']},
        ],
        'preprocess': [
            {'func': pre_elec_price_components, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Stacked Components',{'x':'year','y':'value', 'y_agg':'Weighted Ave', 'y_weight':'load', 'series':'elec_comp_type', 'explode': 'scenario', 'chart_type':'Bar'}),
            ('Scenario Compare',{'x':'year','y':'value', 'y_agg':'Weighted Ave', 'y_weight':'load', 'series':'scenario', 'explode': 'elec_comp_type', 'chart_type':'Line'}),
        )),
        }
    ),
    ('Water Withdrawals (Bil Gals)',
        {'file': "water_output.gdx",
        'param': 'WaterWqctnallyears',
        'columns': ["tech", "cool_tech", "n", "year", "value"],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': 0.001, 'column': 'value'}},
            {'func': add_huc_reg, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Total',{'x':'year','y':'value', 'series':'scenario', 'chart_type':'Line'}),
            ('Explode huc',{'x':'year','y':'value', 'series':'scenario', 'explode':'huc_custom', 'chart_type':'Line'}),
            ('Explode scenario for huc',{'x':'year','y':'value', 'series':'huc_custom', 'explode':'scenario', 'chart_type':'Line'}),
        )),
        }
    ),
    ('Water Consumption (Bil Gals)',
        {'file': "water_output.gdx",
        'param': 'WaterCqctnallyears',
        'columns': ["tech", "cool_tech", "n", "year", "value"],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': 0.001, 'column': 'value'}},
            {'func': add_huc_reg, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Total',{'x':'year','y':'value', 'series':'scenario', 'chart_type':'Line'}),
            ('Custom huc',{'x':'year','y':'value', 'series':'scenario', 'explode':'huc_custom', 'chart_type':'Line'}),
            ('Explode scenario for huc',{'x':'year','y':'value', 'series':'huc_custom', 'explode':'scenario', 'chart_type':'Line'}),
        )),
        }
    ),
    ('Tech Val Streams mps chosen',
        {'file': 'valuestreams/valuestreams_chosen.csv',
        'preprocess': [
            {'func': sum_over_cols, 'args': {'group_cols': ['tech', 'new_old', 'year', 'n', 'type'], 'sum_over_cols': ['m']}},
            {'func': scale_column, 'args': {'scale_factor': 1000*CRF_reeds*inflation_mult/1e9, 'column': 'value'}},
        ],
        'presets': collections.OrderedDict((
            ('New Bil $ by type over time', {'x':'year','y':'value','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'new_old':['new']}}),
            ('Old Bil $ by type over time', {'x':'year','y':'value','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'new_old':['old']}}),
            ('Mixed Bil $ by type over time', {'x':'year','y':'value','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'new_old':['mixed']}}),
            ('Retire Bil $ by type over time', {'x':'year','y':'value','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'new_old':['retire']}}),
        )),
        }
    ),
    ('Tech Val Streams potential',
        {'file': 'valuestreams/valuestreams_potential.csv',
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': inflation_mult, 'column': '$/kW'}},
        ],
        'presets': collections.OrderedDict((
            ('$/kW by type final', {'x':'var_set','y':'$/kW','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9s', 'sync_axes':'No', 'filter': {'year':'last','type':{'exclude':['profit','reduced_cost']},'new_old':['new']}}),
            ('$/kW by type retire final', {'x':'var_set','y':'$/kW','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9s', 'sync_axes':'No', 'filter': {'year':'last','type':{'exclude':['profit','reduced_cost']},'new_old':['retire']}}),
            ('$/kW by type final p60', {'x':'var_set','y':'$/kW','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9s', 'sync_axes':'No', 'filter': {'n':['p60'],'year':'last','type':{'exclude':['profit','reduced_cost']},'new_old':['new']}}),
            ('$/kW by type retire final p60', {'x':'var_set','y':'$/kW','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9s', 'sync_axes':'No', 'filter': {'n':['p60'],'year':'last','type':{'exclude':['profit','reduced_cost']},'new_old':['retire']}}),
        )),
        }
    ),
    ('Tech Val Streams $/MWh potential',
        {'sources': [
            {'name': 'valstream', 'file': 'valuestreams/valuestreams_potential.csv'},
            {'name': 'load', 'file': 'valuestreams/load_pca_potential.csv'},
            {'name': 'prices_nat', 'file': 'MarginalPrices.gdx', 'param': 'pmarg_nat_ann_allyrs', 'columns': ['type','year', '$/MWh']},
        ],
        'preprocess': [
            {'func': pre_tech_val_streams_potential, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('$/MWh by type final', {'x':'var_set','y':'$/kW','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'type', 'adv_col_base':'MWh/kW', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9s', 'sync_axes':'No', 'filter': {'year':'last','type':{'exclude':['profit','reduced_cost']},'new_old':['new']}}),
            ('Stacked Value factor by type final', {'x':'var_set','y':'$/kW','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'type', 'adv_col_base':'block_revenue', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9s', 'sync_axes':'No', 'filter': {'year':'last','type':{'exclude':['MWh/kW','profit','reduced_cost','fix_cost','gp']},'new_old':['new']}}),
            ('$/MWh by type final p60', {'x':'var_set','y':'$/kW','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'type', 'adv_col_base':'MWh/kW', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9s', 'sync_axes':'No', 'filter': {'n':['p60'],'year':'last','type':{'exclude':['profit','reduced_cost']},'new_old':['new']}}),
            ('Stacked Value factor by type final p60', {'x':'var_set','y':'$/kW','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'type', 'adv_col_base':'block_revenue', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9s', 'sync_axes':'No', 'filter': {'n':['p60'],'year':'last','type':{'exclude':['MWh/kW','profit','reduced_cost','fix_cost','gp']},'new_old':['new']}}),
        )),
        }
    ),
    ('Stacked profitability potential',
        {'file': 'valuestreams/valuestreams_potential.csv',
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': inflation_mult, 'column': '$/kW'}},
            {'func': pre_stacked_profitability_potential, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Stacked profitability final', {'x':'var_set','y':'$/kW','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'adv_op':'Ratio', 'adv_col':'type', 'adv_col_base':'cost', 'plot_width':'1200', 'bar_width':'0.9s', 'sync_axes':'No', 'filter': {'year':'last','type':{'exclude':['profit','reduced_cost']},'new_old':['new']}}),
            ('Stacked profitability final p60', {'x':'var_set','y':'$/kW','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'adv_op':'Ratio', 'adv_col':'type', 'adv_col_base':'cost', 'plot_width':'1200', 'bar_width':'0.9s', 'sync_axes':'No', 'filter': {'n':['p60'],'year':'last','type':{'exclude':['profit','reduced_cost']},'new_old':['new']}}),
        )),
        }
    ),
    ('Marginal Prices',
        {'file': 'MarginalPrices.gdx',
        'param': 'pmarg_BA_allyrs',
        'columns': ['n', 'm', 'type', 'year', 'value'],
        }
    ),
    ('Nat Ann Marginal Prices',
        {'file': 'MarginalPrices.gdx',
        'param': 'pmarg_nat_ann_allyrs',
        'columns': ['type', 'year', 'value'],
        }
    ),
    ('JEDI Outputs',
        {'file': "JEDI_out.gdx",
        'param': 'JEDI',
        'columns': ["jedi_scenario", "jedi_tech", "state_plus_dc", "category", "metric", "directness", "year", "value"],
        'index': ['jedi_scenario', 'jedi_tech', 'state_plus_dc', 'category', 'metric', 'directness', 'year'],
        'preprocess': [
            {'func': scale_column_filtered, 'args': {'by_column': 'metric', 'by_vals': ['jobs'], 'change_column': 'value', 'scale_factor': .000001}},
            {'func': scale_column_filtered, 'args': {'by_column': 'metric', 'by_vals': ['earnings','output','value_add'], 'change_column': 'value', 'scale_factor': .001}},
        ],
        'presets': collections.OrderedDict((
            ('Main Metrics Lines',{'x':'year','y':'value', 'series':'scenario', 'explode':'metric', 'explode_group':'jedi_scenario', 'chart_type':'Line', 'sync_axes':'No'}),
            ('Total Jobs',{'x':'year','y':'value', 'series':'scenario', 'explode':'jedi_scenario', 'chart_type':'Line', 'filter': {'metric':['jobs']}}),
            ('Stacked Jobs By Tech',{'x':'year','y':'value', 'series':'jedi_tech', 'explode':'scenario', 'explode_group':'jedi_scenario', 'chart_type':'Bar', 'filter': {'metric':['jobs']}}),
            ('Stacked Jobs By Directness',{'x':'year','y':'value', 'series':'directness', 'explode':'scenario', 'explode_group':'jedi_scenario', 'chart_type':'Bar', 'filter': {'metric':['jobs']}}),
            ('Stacked Jobs By Directness Explode Tech',{'x':'year','y':'value', 'series':'directness', 'explode':'scenario', 'explode_group':'jedi_tech', 'chart_type':'Bar', 'filter': {'metric':['jobs']}}),
            ('Stacked Jobs By Category',{'x':'year','y':'value', 'series':'category', 'explode':'scenario', 'explode_group':'jedi_scenario', 'chart_type':'Bar', 'filter': {'metric':['jobs']}}),
            ('Stacked Jobs By Category Explode Tech',{'x':'year','y':'value', 'series':'category', 'explode':'scenario', 'explode_group':'jedi_tech', 'chart_type':'Bar', 'filter': {'metric':['jobs']}}),
            ('Jobs By Tech',{'x':'year','y':'value', 'series':'scenario', 'explode':'jedi_tech', 'explode_group':'jedi_scenario', 'chart_type':'Line', 'filter': {'metric':['jobs']}}),
            ('Jobs By Directness',{'x':'year','y':'value', 'series':'scenario', 'explode':'directness', 'explode_group':'jedi_scenario', 'chart_type':'Line', 'filter': {'metric':['jobs']}}),
            ('Stacked Earnings By Tech',{'x':'year','y':'value', 'series':'jedi_tech', 'explode':'scenario', 'explode_group':'jedi_scenario', 'chart_type':'Bar', 'filter': {'metric':['earnings']}}),
            ('Earnings By Tech',{'x':'year','y':'value', 'series':'scenario', 'explode':'jedi_tech', 'explode_group':'jedi_scenario', 'chart_type':'Line', 'filter': {'metric':['earnings']}}),
            ('Stacked Output By Tech',{'x':'year','y':'value', 'series':'jedi_tech', 'explode':'scenario', 'explode_group':'jedi_scenario', 'chart_type':'Bar', 'filter': {'metric':['output']}}),
            ('Output By Tech',{'x':'year','y':'value', 'series':'scenario', 'explode':'jedi_tech', 'explode_group':'jedi_scenario', 'chart_type':'Line', 'filter': {'metric':['output']}}),
            ('Stacked Value Add By Tech',{'x':'year','y':'value', 'series':'jedi_tech', 'explode':'scenario', 'explode_group':'jedi_scenario', 'chart_type':'Bar', 'filter': {'metric':['value_add']}}),
            ('Value Add By Tech',{'x':'year','y':'value', 'series':'scenario', 'explode':'jedi_tech', 'explode_group':'jedi_scenario', 'chart_type':'Line', 'filter': {'metric':['value_add']}}),
            ('Average Onsite Jobs Map 2017-end',{'chart_type':'Map', 'x':'st', 'y':'value', 'explode':'scenario', 'explode_group':'jedi_scenario', 'y_scale':'29412', 'filter': {'metric':['jobs'], 'directness':['Onsite'], 'year': {'start':2017},}}), #y_scale, 29412 = 1000000(jobs per mil jobs) / 34(total years), so we end up with avarage jobs
        )),
        }
    ),
    ('Marginal Capacity Value',
        {'sources': [
            {'name': 'new_cap', 'file': 'CONVqn.gdx', 'param': 'CONVqn_newallyears', 'columns': ['tech', 'n', 'year', 'Capacity (GW)']},
            {'name': 'cv_mar', 'file': 'Reporting.gdx', 'param': 'CVmar_annual_average', 'columns': ['tech', 'n', 'year', 'Capacity Value']},
        ],
        'preprocess': [
            {'func': pre_marginal_capacity_value, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Marginal Capacity Value',{'chart_type':'Line', 'x':'year', 'y':'Capacity Value', 'y_agg':'Weighted Ave', 'y_weight':'Capacity (GW)', 'series':'scenario', 'explode':'tech', 'filter': {}}),
        )),
        }
    ),
    ('Marginal Curtailment',
        {'file': "Reporting.gdx",
        'param': 'VRREOut',
        'columns': ["n","m","year","rtech","type","value"],
        'preprocess': [
            {'func': pre_marginal_curtailment_filter, 'args': {}},
            {'func': sum_over_cols, 'args': {'group_cols': ['n', 'year', 'rtech', 'type'], 'sum_over_cols': ['m']}},
            {'func': pre_marginal_curtailment, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Marginal Curtailment by tech', {'chart_type':'Line', 'x':'year', 'y':'surpmar', 'y_agg':'Weighted Ave', 'y_weight':'gen', 'series':'scenario', 'explode':'rtech', 'filter': {}}),
        )),
        }
    ),
    ('Marginal Curtailment m',
        {'file': "Reporting.gdx",
        'param': 'VRREOut',
        'columns': ["n","m","year","rtech","type","value"],
        'preprocess': [
            {'func': pre_marginal_curtailment, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Marginal Curtailment by tech', {'chart_type':'Line', 'x':'year', 'y':'surpmar', 'y_agg':'Weighted Ave', 'y_weight':'gen', 'series':'scenario', 'explode':'rtech', 'filter': {}}),
        )),
        }
    ),
    ('wat_access',
        {'file': "water_output.gdx",
        'param': 'WatAccessallyears',
        'columns': ["n", "class", "year", "value"],
        }
    ),
    ('<Old> System Cost',
        {'file': 'Reporting.gdx',
        'param': 'aSystemCost',
        'columns': ['cost_cat', 'year', 'Cost (2015$)'],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': inflation_mult, 'column': 'Cost (2015$)'}},
            {'func': discount_costs, 'args': {}},
        ],
        }
    ),
    ('cap_wind_nrr',
        {'file': 'CONVqn.gdx',
        'param': 'WR2GOallyears',
        'columns': ["i",  "class",  "windtype", "bin","year", "value"],
        }
    ),
    ('cap_csp_nrr',
        {'file': 'CONVqn.gdx',
        'param': 'CSP2GOallyears',
        'columns': ["i",  "cspclass", "bin","year", "value"],
        }
    ),
    ('cap_upv_nrr',
        {'file': "CONVqn.gdx",
        'param': 'UPVR2GOallyears',
        'columns': ["n",  "upvclass", "bin","year", "value"],
        }
    ),
    ('cap_dupv_nrr',
        {'file': "CONVqn.gdx",
        'param': 'DUPVR2GOallyears',
        'columns': ["n",  "dupvclass", "bin","year", "value"],
        }
    ),
    ('op_cap',
        {'file': "CONVqn.gdx",
        'param': 'OperCONVqnallyears',
        'columns': ["tech", "n", "year", "value"],
        }
    ),
    ('rebuild',
        {'file': "CONVqn.gdx",
        'param': 'Rebuildqnallyears',
        'columns': ["tech", "n", "year", "value"],
        }
    ),
    ('upgrade',
        {'file': "CONVqn.gdx",
        'param': 'Upgradeqnallyears',
        'columns': ["techold", "technew", "n", "year", "value"],
        }
    ),
    ('hydb',
        {'file': "CONVqn.gdx",
        'param': 'HydBin_allyrs',
        'columns': ["cat", "class", "n", "year", "value"],
        }
    ),
    ('pshb',
        {'file': "CONVqn.gdx",
        'param': 'PHSBIN_allyrs',
        'columns': ["class", "n", "year", "value"],
        }
    ),
    ('plan_res',
        {'file': "Reporting.gdx",
        'param': 'PlanRes',
        'columns': ["tech", "n", "year", "m", "value"],
        }
    ),
    ('oper_res',
        {'file': "Reporting.gdx",
        'param': 'OperRes',
        'columns': ["tech", "n", "year", "m", "value"],
        }
    ),
    ('vrre',
        {'file': "Reporting.gdx",
        'param': 'VRREOut',
        'columns': ["n","m","year","tech","type","value"],
        }
    ),
    ('trans',
        {'file': "Reporting.gdx",
        'param': 'Transmission',
        'columns': ["n", "n2", "year", "type", "value"],
        }
    ),
    ('annual_rep',
        {'file': "Reporting.gdx",
        'param': 'AnnualReport',
        'columns': ["n", "year", "type", "value"],
        }
    ),
    ('annual_rep',
        {'file': "Reporting.gdx",
        'param': 'AnnualReport',
        'columns': ["n", "year", "type", "value"],
        }
    ),
    ('fuel_cost',
        {'file': "Reporting.gdx",
        'param': 'fuelcost',
        'columns': ["year", "country", "type", "value"],
        }
    ),
    ('reg_fuel_cost',
        {'file': "Reporting.gdx",
        'param': 'reg_fuelcost',
        'columns': ["nerc", "year", "type", "value"],
        }
    ),
    ('switches',
        {'file': "Reporting.gdx",
        'param': 'ReportSwitches',
        'columns': ["class", "switch", "value", "ignore"],
        }
    ),
    ('st_rps',
        {'file': "StRPSoutputs.gdx",
        'param': 'StRPSoutput',
        'columns': ["tech", "st", "year", "value"],
        }
    ),
    ('st_rps_mar',
        {'file': "StRPSoutputs.gdx",
        'param': 'StRPSmarginalout',
        'columns': ["st", "year", "const", "value"],
        }
    ),
    ('st_rps_rec',
        {'file': "StRPSoutputs.gdx",
        'param': 'RECallyears',
        'columns': ["n", "st2", "tech", "year", "value"],
        }
    ),
    ('ac_flow',
        {'file': "Transmission.gdx",
        'param': 'TransFlowAC',
        'columns': ["n", "n2", "year", "m", "value"],
        }
    ),
    ('dc_flow',
        {'file': "Transmission.gdx",
        'param': 'TransFlowDC',
        'columns': ["n", "n2", "year", "m", "value"],
        }
    ),
    ('cont_flow',
        {'file': "Transmission.gdx",
        'param': 'ContractFlow',
        'columns': ["n", "n2", "year", "m", "value"],
        }
    ),
    ('obj_fnc',
        {'file': "z.gdx",
        'param': 'z_allyrs',
        'columns': ["year", "value"],
        }
    ),
))
