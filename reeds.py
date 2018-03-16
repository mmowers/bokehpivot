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
def get_pv_mult(year, type, dinvest=0.054439024, dsocial=0.03, lifetime=20, refyear=2017, lastyear=2050):
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

def pre_value_factors(dfs, **kw):
    #start with dfs['gen'], and expand to include all combinations of tech, n, year, m
    df = dfs['gen']
    idx_cols = ['tech','n','year','m']
    full_idx = pd.MultiIndex.from_product([df[col].unique().tolist() for col in idx_cols], names=idx_cols)
    df = df.set_index(idx_cols).reindex(full_idx).reset_index()
    #load hours.csv
    hours = pd.read_csv(this_dir_path + '/in/hours.csv')
    #merge hours into df
    df = pd.merge(left=df, right=hours, on='m', sort=False)
    #convert gen from MW to MWh
    df['Gen (MWh)'] = df['Gen (MWh)']*df['hours']
    #remove all but load_pca from dfs['load_marg']
    df_lm = dfs['load_marg']
    df_lm = df_lm[df_lm['type'] == 'load_pca'].copy()
    #apply inflation
    df_lm['Price ($/MWh)'] = df_lm['Price ($/MWh)'] * inflation_mult
    #drop 'type' column
    df_lm.drop('type', axis='columns', inplace=True)
    #merge load_marg into df
    df = pd.merge(left=df, right=df_lm, on=['n','m','year'], how='left', sort=False)
    return df

def pre_value_streams(df, **kw):
    #Separate marginals and quantities into separate columns and add load_marginal ($/MWh) and load (MWh) columns

    #First, fill missing data with 0 by pivoting out and melting back, so that everywhere we have load quantities,
    #we'll have marginals and values for all our constraints. Without this we won't get the right weighted averages later.
    val_stream_types = df['val_stream_type'].unique().tolist()
    pivot_index = [i for i in df.columns if i not in ['val_stream_type', 'value']]
    df = df.pivot_table(index=pivot_index, columns='val_stream_type', values='value').reset_index()
    df.columns.name = None
    df = pd.melt(df, id_vars=pivot_index, value_vars=val_stream_types, var_name='val_stream_type', value_name= 'value')
    df['value'] = df['value'].fillna(0)
    #Pivot out by val_out_type so that marginal and quantity are separate columns
    pivot_index = [i for i in df.columns if i not in ['val_out_type', 'value']]
    df = df.pivot_table(index=pivot_index, columns='val_out_type', values='value').reset_index()
    df.columns.name = None
    #adjust marginals by inflation
    df['marginal'] = df['marginal'] * inflation_mult
    #Find the load quantities and store in separate dataframe
    df_load = df[(df['val_stream_type']=='load_pca')].copy()
    df_load.drop(['val_stream_type','marginal'], axis='columns', inplace=True)
    #Now merge load quantities back into dataframe
    merge_index = [i for i in df.columns if i not in ['val_stream_type', 'marginal', 'quantity']]
    df = pd.merge(left=df, right=df_load, how='outer', on=merge_index, sort=False)
    df.rename(columns={'quantity_x': 'quantity', 'quantity_y': 'MWh'}, inplace=True)
    df['$/MWh'] = df['quantity']*df['marginal']/df['MWh']
    df['Bil $'] = df['quantity']*df['marginal']/1e9
    return df

def pre_tech_value_streams(df, **kw):
    #Get quantity into a separate column, then add $/MWh column.
    #First, separate quantities into their own dataframe.
    df_quant = df[df['tech_val_type']=='quantity'].copy()
    df_quant.drop('tech_val_type', axis='columns', inplace=True)
    df = df[df['tech_val_type']!='quantity'].copy()
    #apply inflation to all values/costs.
    df['value'] = df['value'] * inflation_mult
    tech_val_types = df['tech_val_type'].unique().tolist()
    merge_index = [i for i in df.columns if i not in ['tech_val_type', 'value']]
    #before merging with quantity, make sure we have values for all val stream types wherever we have costs.
    #Without this we won't get the right weighted averages later.
    df = df.pivot_table(index=merge_index, columns='tech_val_type', values='value').reset_index()
    df.columns.name = None
    df = pd.melt(df, id_vars=merge_index, value_vars=tech_val_types, var_name='tech_val_type', value_name= 'value')
    df['value'] = df['value'].fillna(0)
    #If we are calculating total, sum the desired tech_val_types and remove the others
    if 'opt' in kw and kw['opt'] == 'tot':
        df = df[df['tech_val_type'].isin(['load_pca','res_marg','oper_res'])]
        df = sum_over_cols(df, sum_over_cols='tech_val_type', group_cols=merge_index)
    #Now do the merge.
    df = pd.merge(left=df, right=df_quant, how='outer', on=merge_index, sort=False)
    df.rename(columns={'value_x': '$', 'value_y': 'MWh'}, inplace=True)
    df['$/MWh'] = df['$']/df['MWh']
    df['$'] = df['$']/1e9
    df.rename(columns={'$':'Bil $'}, inplace=True)
    return df

def pre_revenue_streams(dfs, **kw):
    #This function will add revenue streams for a hypothetical block generation tech for comparison to the real tech
    #The hypothetical block gen tech has the same costs except no trans costs, and only has load_pca and res_marg value streams.
    #Load_pca and res_marg value streams are calculated by applying average annual prices of load_pca and res_marg 
    #start with dfs['val_streams']
    df_val_streams = dfs['val_streams']
    #remove rps value stream
    df_val_streams = df_val_streams[~df_val_streams['tech_val_type'].isin(['rps'])]
    #remove existing and delete new_exist column
    df_val_streams = df_val_streams[df_val_streams['new_exist'].isin(['new'])]
    df_val_streams.drop('new_exist', axis='columns', inplace=True)
    #sum over timeslices and call pre_tech_value_streams
    df_val_streams = sum_over_cols(df_val_streams, sum_over_cols='m', group_cols=['tech', 'n', 'year', 'tech_val_type'])
    df_val_streams = pre_tech_value_streams(df_val_streams)
    #Add column for real vs hypothetical
    df_val_streams['hypo_type'] = 'real'
    #Now create the hypothetical block generation tech with adjusted load_pca and res_marg streams
    #copy df_val_streams into df_block and change type to block. Also copy into df_block_load_dist,
    #the load-distributed block
    df_block = df_val_streams.copy()
    df_block_load_dist = df_val_streams.copy()
    df_block['hypo_type'] = 'block'
    df_block_load_dist['hypo_type'] = 'block_load_dist'
    #remove 'oper_res', 'trans', 'other' categories from df_block because these value streams are not considered for the hypothetical block gen tech.
    df_block = df_block[~df_block['tech_val_type'].isin(['oper_res', 'trans', 'other'])]
    df_block_load_dist = df_block_load_dist[~df_block_load_dist['tech_val_type'].isin(['oper_res', 'trans', 'other'])]
    #Read in dfs['prices'], the average annual prices in $/MWh
    df_price = dfs['prices']
    df_price_load_dist = dfs['prices_nat']
    #remove all but load_pca and res_marg
    df_price = df_price[df_price['type'].isin(['load_pca','res_marg'])].copy() #copy() is used to prevent SettingWithCopyWarning
    df_price_load_dist = df_price_load_dist[df_price_load_dist['type'].isin(['load_pca','res_marg'])].copy() #copy() is used to prevent SettingWithCopyWarning
    #rename type to tech_val_type
    df_price.rename(columns={'type':'tech_val_type'}, inplace=True)
    df_price_load_dist.rename(columns={'type':'tech_val_type'}, inplace=True)
    #adjust for inflation
    df_price['price'] = df_price['price'] * inflation_mult
    df_price_load_dist['price'] = df_price_load_dist['price'] * inflation_mult
    #Merge df_price into df_block
    df_block = pd.merge(left=df_block, right=df_price, on=['n','year', 'tech_val_type'], how='left', sort=False)
    df_block_load_dist = pd.merge(left=df_block_load_dist, right=df_price_load_dist, on=['year', 'tech_val_type'], how='left', sort=False)
    df_blocks = pd.concat([df_block, df_block_load_dist], ignore_index=True)
    #Calculate block gen revenue for load_pca and res_marg by multiplying MWh by average prices.
    #df_block assumes that hypothetical block generator is in same n as the real generator.
    #df_block_load_dist assumes that hypothetical block generator is load-distributed.
    df_block_cond = df_blocks['tech_val_type'].isin(['load_pca','res_marg'])
    df_blocks.loc[df_block_cond, '$/MWh'] = df_blocks.loc[df_block_cond, 'price']
    df_blocks.loc[df_block_cond, 'Bil $'] = df_blocks.loc[df_block_cond, 'MWh'] * df_blocks.loc[df_block_cond, 'price']/1e9
    #concatenate df_block with df_val_streams
    df = pd.concat([df_val_streams, df_blocks], ignore_index=True)
    return df

def pre_profitability_index(df, **kw):
    #This preprocess calculates a profitability index column, defined as system value / cost
    costs = ['water','varom','trans','incent','fuel','fixom','capital']
    #leaving out rps in system_vals because we are implementing forcing functions
    system_vals = ['load_pca','res_marg','oper_res','other','surplus']
    df = df[df['tech_val_type'].isin(costs + system_vals + ['quantity'])].copy()
    df['agg_cat'] = 'energy'
    df.loc[df['tech_val_type'].isin(system_vals), 'agg_cat'] = 'val'
    df_costs = df['tech_val_type'].isin(costs)
    df.loc[df_costs, 'agg_cat'] = 'cost'
    #costs are negative so we must multiply by -1
    df.loc[df_costs, 'value'] = df.loc[df_costs, 'value'] * -1
    df.drop(['tech_val_type'], axis='columns', inplace=True)
    #add up all costs and values
    df =  df.groupby(['tech', 'new_exist', 'year', 'n','agg_cat'], sort=False, as_index =False).sum()
    #now turn cost, val, and energy into separate columns
    df = df.pivot_table(index=['tech', 'new_exist', 'year', 'n'], columns='agg_cat', values='value').reset_index()
    df['cost'] = df['cost'] * inflation_mult
    df['val'] = df['val'] * inflation_mult
    df['profit_index'] = df['val'] / df['cost']
    df['LCOV'] = df['cost'] / df['val']
    df['LCOE'] = df['cost'] / df['energy']
    df['LCOV/LCOE'] = df['LCOV'] / df['LCOE']
    return df

def pre_stacked_profitability(df, **kw):
    #Sum all costs so that we can calculate value / total cost for each value stream
    #remove quantity
    df = df[df['tech_val_type'] != 'quantity'].copy()
    #adjust by inflation and express as billion $
    df['Bil $'] = df['Bil $'] * inflation_mult / 1e9
    #label all costs the same so they can be grouped
    costs = ['water','varom','trans','incent','fuel','fixom','capital']
    df.loc[df['tech_val_type'].isin(costs),'tech_val_type'] = 'cost'
    df.loc[df['tech_val_type'] == 'cost','Bil $'] *= -1
    #sum costs
    df =  df.groupby(['tech', 'new_exist', 'year', 'n','tech_val_type'], sort=False, as_index =False).sum()
    return df

def pre_stacked_profitability_potential(df, **kw):
    #Sum all costs so that we can calculate value / total cost for each value stream
    #remove quantity
    #label all costs the same so they can be grouped
    costs = ['fix cost','var cost','trans cost','gp']
    df.loc[df['type'].isin(costs),'type'] = 'cost'
    df.loc[df['type'] == 'cost','value_per_unit'] *= -1
    #sum costs
    df =  df.groupby(['tech', 'year', 'n','type','var_set'], sort=False, as_index =False).sum()
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
    ('Value Streams',
        {'file': 'valuestreams.gdx',
        'param': 'val_streams',
        'columns': ['val_stream_type', 'year', 'n', 'm', 'val_out_type', 'value'],
        'index': ['val_stream_type', 'year', 'n', 'm'],
        'preprocess': [
            {'func': pre_value_streams, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Bil $ by type over time', {'x':'year','y':'Bil $','y_agg':'Sum','series':'val_stream_type','explode': 'scenario', 'chart_type':'Bar', 'bar_width':'1.75',}),
            ('$/MWh by type over time', {'x':'year','y':'$/MWh','y_agg':'Weighted Ave', 'y_weight':'MWh','series':'val_stream_type','explode': 'scenario', 'chart_type':'Bar', 'bar_width':'1.75',}),
            ('Marginal Prices', {'x':'year','y':'marginal','y_agg':'Weighted Ave', 'y_weight':'quantity','explode':'val_stream_type','series': 'scenario', 'chart_type':'Line', 'sync_axes':'No',}),
            ('Final $/MWh by type by timeslice, custreg', {'chart_type':'Bar', 'x':'custreg', 'y':'$/MWh', 'y_agg':'Weighted Ave', 'y_weight':'MWh', 'series':'val_stream_type', 'explode':'scenario', 'explode_group':'m', 'filter': {'year':'last', }}),
            ('Final State map Load ($/MWh)', {'chart_type':'Map', 'x':'st', 'y':'$/MWh', 'y_agg':'Weighted Ave', 'y_weight':'MWh', 'explode':'scenario', 'filter': {'val_stream_type':['Energy Demand'], 'year':'last', }}),
            ('Final State map by timeslice ($/MWh)', {'chart_type':'Map', 'x':'st', 'y':'$/MWh', 'y_agg':'Weighted Ave', 'y_weight':'MWh', 'explode':'scenario', 'explode_group':'m', 'filter': {'val_stream_type':['Energy Demand'], 'year':'last', }}),
        )),
        }
    ),
    ('Tech Value Streams',
        {'file': 'valuestreams.gdx',
        'param': 'tech_val_streams',
        'columns': ['tech', 'new_exist', 'year', 'n', 'm', 'tech_val_type', 'value'],
        'preprocess': [
            {'func': sum_over_cols, 'args': {'group_cols': ['tech', 'new_exist', 'year', 'tech_val_type'], 'sum_over_cols': ['m', 'n']}},
            {'func': pre_tech_value_streams, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('New $/MWh by type over time', {'x':'year','y':'$/MWh','y_agg':'Weighted Ave', 'y_weight':'MWh','series':'tech_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'filter': {'new_exist':['new']}}),
            ('Bil $ by type over time', {'x':'year','y':'Bil $','series':'tech_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'filter': {'new_exist':['new']}}),
        )),
        }
    ),
    ('Tech Value Streams n',
        {'file': 'valuestreams.gdx',
        'param': 'tech_val_streams',
        'columns': ['tech', 'new_exist', 'year', 'n', 'm', 'tech_val_type', 'value'],
        'preprocess': [
            {'func': sum_over_cols, 'args': {'group_cols': ['tech', 'new_exist', 'year', 'n', 'tech_val_type'], 'sum_over_cols': ['m']}},
            {'func': pre_tech_value_streams, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('New $/MWh by type over time', {'x':'year','y':'$/MWh','y_agg':'Weighted Ave', 'y_weight':'MWh','series':'tech_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'filter': {'new_exist':['new']}}),
            ('Bil $ by type over time', {'x':'year','y':'Bil $','series':'tech_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'filter': {'new_exist':['new']}}),
        )),
        }
    ),
    ('Tech Value Streams n,m',
        {'file': 'valuestreams.gdx',
        'param': 'tech_val_streams',
        'columns': ['tech', 'new_exist', 'year', 'n', 'm', 'tech_val_type', 'value'],
        'preprocess': [
            {'func': pre_tech_value_streams, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('New $/MWh by type over time', {'x':'year','y':'$/MWh','y_agg':'Weighted Ave', 'y_weight':'MWh','series':'tech_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'filter': {'new_exist':['new']}}),
            ('Bil $ by type over time', {'x':'year','y':'Bil $','series':'tech_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'filter': {'new_exist':['new']}}),
        )),
        }
    ),
    ('Tech Value Streams 2',
        {'file': 'valuestreams.gdx',
        'param': 'tech_val_streams_2',
        'columns': ['tech', 'new_exist', 'year', 'n', 'm', 'tech_val_type', 'value'],
        'preprocess': [
            {'func': sum_over_cols, 'args': {'group_cols': ['tech', 'new_exist', 'year', 'tech_val_type'], 'sum_over_cols': ['m', 'n']}},
            {'func': pre_tech_value_streams, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('New $/MWh by type over time', {'x':'year','y':'$/MWh','y_agg':'Weighted Ave', 'y_weight':'MWh','series':'tech_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'filter': {'new_exist':['new']}}),
            ('Bil $ by type over time', {'x':'year','y':'Bil $','series':'tech_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'filter': {'new_exist':['new']}}),
        )),
        }
    ),
    ('Tech Value Streams 2 n,m',
        {'file': 'valuestreams.gdx',
        'param': 'tech_val_streams_2',
        'columns': ['tech', 'new_exist', 'year', 'n', 'm', 'tech_val_type', 'value'],
        'preprocess': [
            {'func': pre_tech_value_streams, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('New $/MWh by type over time', {'x':'year','y':'$/MWh','y_agg':'Weighted Ave', 'y_weight':'MWh','series':'tech_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'filter': {'new_exist':['new']}}),
            ('Bil $ by type over time', {'x':'year','y':'Bil $','series':'tech_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'filter': {'new_exist':['new']}}),
        )),
        }
    ),
    ('Tech Value Streams 3',
        {'file': 'valuestreams.gdx',
        'param': 'tech_val_streams_3',
        'columns': ['tech', 'new_exist', 'year', 'n', 'm', 'tech_val_type', 'value'],
        'preprocess': [
            {'func': sum_over_cols, 'args': {'group_cols': ['tech', 'new_exist', 'year', 'tech_val_type'], 'sum_over_cols': ['m', 'n']}},
            {'func': pre_tech_value_streams, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('New $/MWh by type over time', {'x':'year','y':'$/MWh','y_agg':'Weighted Ave', 'y_weight':'MWh','series':'tech_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'filter': {'new_exist':['new']}}),
            ('Bil $ by type over time', {'x':'year','y':'Bil $','series':'tech_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'filter': {'new_exist':['new']}}),
        )),
        }
    ),
    ('Tech Value Streams 3 n,m',
        {'file': 'valuestreams.gdx',
        'param': 'tech_val_streams_3',
        'columns': ['tech', 'new_exist', 'year', 'n', 'm', 'tech_val_type', 'value'],
        'preprocess': [
            {'func': pre_tech_value_streams, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('New $/MWh by type over time', {'x':'year','y':'$/MWh','y_agg':'Weighted Ave', 'y_weight':'MWh','series':'tech_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'filter': {'new_exist':['new']}}),
            ('Bil $ by type over time', {'x':'year','y':'Bil $','series':'tech_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'filter': {'new_exist':['new']}}),
        )),
        }
    ),
    ('Tech Value Streams Tot',
        {'file': 'valuestreams.gdx',
        'param': 'tech_val_streams_3',
        'columns': ['tech', 'new_exist', 'year', 'n', 'm', 'tech_val_type', 'value'],
        'preprocess': [
            {'func': sum_over_cols, 'args': {'group_cols': ['tech', 'new_exist', 'year', 'tech_val_type'], 'sum_over_cols': ['m', 'n']}},
            {'func': pre_tech_value_streams, 'args': {'opt':'tot'}},
        ],
        'presets': collections.OrderedDict((
            ('New $/MWh over time', {'x':'year','y':'$/MWh','y_agg':'Weighted Ave', 'y_weight':'MWh','series':'scenario', 'explode': 'tech', 'chart_type':'Line', 'filter': {'new_exist':['new']}}),
            ('Bil $ over time', {'x':'year','y':'Bil $','series':'scenario', 'explode': 'tech', 'chart_type':'Line', 'filter': {'new_exist':['new']}}),
        )),
        }
    ),
    ('Revenue Streams',
        {'sources': [
            {'name': 'val_streams', 'file': 'valuestreams.gdx', 'param': 'tech_val_streams_3', 'columns': ['tech', 'new_exist', 'year', 'n', 'm', 'tech_val_type', 'value']},
            {'name': 'prices', 'file': 'MarginalPrices.gdx', 'param': 'pmarg_BA_ann_allyrs', 'columns': ['n', 'type','year', 'price']},
            {'name': 'prices_nat', 'file': 'MarginalPrices.gdx', 'param': 'pmarg_nat_ann_allyrs', 'columns': ['type','year', 'price']},
        ],
        'preprocess': [
            {'func': pre_revenue_streams, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('$/MWh total revenue', {'x':'year','y':'$/MWh','y_agg':'Weighted Ave', 'y_weight':'MWh','series':'hypo_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Line', 'filter': {'tech_val_type': ['Energy Value', 'Capacity Value', 'Ancillary Value', 'Other']}}),
            ('Wind $/MWh', {'x':'year','y':'$/MWh','y_agg':'Weighted Ave', 'y_weight':'MWh','series':'tech_val_type', 'explode': 'hypo_type', 'explode_group': 'scenario', 'chart_type':'Bar', 'bar_width':'1.75', 'filter': {'tech':['Wind']}}),
            ('Value factor', {'x':'year','y':'Bil $','series':'tech', 'explode': 'scenario', 'explode_group': 'hypo_type', 'chart_type':'Line', 'adv_op':'Ratio','adv_col':'hypo_type','adv_col_base':'block', 'filter': {'tech_val_type': ['Energy Value', 'Capacity Value', 'Ancillary Value', 'Trans Cost']}}),
            ('Value factor load dist', {'x':'year','y':'Bil $','series':'tech', 'explode': 'scenario', 'explode_group': 'hypo_type', 'chart_type':'Line', 'adv_op':'Ratio','adv_col':'hypo_type','adv_col_base':'block_load_dist', 'filter': {'tech_val_type': ['Energy Value', 'Capacity Value', 'Ancillary Value', 'Trans Cost']}}),
        )),
        }
    ),
    ('System Profitability Index',
        {'file': 'valuestreams.gdx',
        'param': 'tech_val_streams_3',
        'columns': ['tech', 'new_exist', 'year', 'n', 'm', 'tech_val_type', 'value'],
        'preprocess': [
            {'func': sum_over_cols, 'args': {'group_cols': ['tech', 'new_exist', 'year', 'n', 'tech_val_type'], 'sum_over_cols': ['m']}},
            {'func': pre_profitability_index, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Profit Index by tech', {'chart_type':'Line', 'x':'year', 'y':'profit_index', 'y_agg':'Weighted Ave', 'y_weight':'cost', 'series':'scenario', 'explode':'tech', 'filter': {}}),
            ('LCOV by tech', {'chart_type':'Line', 'x':'year', 'y':'LCOV', 'y_agg':'Weighted Ave', 'y_weight':'val', 'series':'scenario', 'explode':'tech', 'filter': {}}),
            ('LCOE by tech', {'chart_type':'Line', 'x':'year', 'y':'LCOE', 'y_agg':'Weighted Ave', 'y_weight':'energy', 'series':'scenario', 'explode':'tech', 'filter': {}}),
            ('LCOV/LCOE by tech', {'chart_type':'Line', 'x':'year', 'y':'LCOV/LCOE', 'y_agg':'Weighted Ave', 'y_weight':'LCOE', 'series':'scenario', 'explode':'tech', 'filter': {}}),
        )),
        }
    ),
    ('Stacked System Profitability Index',
        {'file': 'valuestreams.gdx',
        'param': 'tech_val_streams_3',
        'columns': ['tech', 'new_exist', 'year', 'n', 'm', 'tech_val_type', 'Bil $'],
        'preprocess': [
            {'func': sum_over_cols, 'args': {'group_cols': ['tech', 'new_exist', 'year', 'n', 'tech_val_type'], 'sum_over_cols': ['m']}},
            {'func': pre_stacked_profitability, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Stacked Bars', {'chart_type':'Bar', 'x':'year', 'y':'Bil $', 'series':'tech_val_type', 'explode':'scenario', 'explode_group':'tech', 'adv_op':'Ratio', 'adv_col':'tech_val_type', 'adv_col_base':'Cost', 'bar_width':r'1.75', 'filter': {'new_exist':['new']}}),
        )),
        }
    ),
    ('Value Factors',
        {'sources': [
            {'name': 'gen', 'file': 'CONVqn.gdx', 'param': 'CONVqmnallm', 'columns': ['tech', 'n', 'year', 'm', 'Gen (MWh)']},
            {'name': 'load_marg', 'file': 'MarginalPrices.gdx', 'param': 'pmarg_BA_allyrs', 'columns': ['n', 'm', 'type','year', 'Price ($/MWh)']},
        ],
        'preprocess': [
            {'func': pre_value_factors, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Tech Value Factors',{'chart_type':'Line', 'x':'year', 'y':'Price ($/MWh)', 'y_agg':'Weighted Ave Ratio', 'y_weight':'Gen (MWh)', 'y_weight_denom':'hours', 'series':'tech', 'explode':'scenario', 'filter': {}}),
        )),
        }
    ),
    ('New Tech Value Factors',
        {'sources': [
            {'name': 'gen', 'file': 'valuestreams.gdx', 'param': 'gross_gen_new', 'columns': ['tech', 'n', 'year', 'm', 'Gen (MWh)']},
            {'name': 'load_marg', 'file': 'MarginalPrices.gdx', 'param': 'pmarg_BA_allyrs', 'columns': ['n', 'm', 'type','year', 'Price ($/MWh)']},
        ],
        'preprocess': [
            {'func': pre_value_factors, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Tech Value Factors',{'chart_type':'Line', 'x':'year', 'y':'Price ($/MWh)', 'y_agg':'Weighted Ave Ratio', 'y_weight':'Gen (MWh)', 'y_weight_denom':'hours', 'series':'tech', 'explode':'scenario', 'filter': {}}),
        )),
        }
    ),
    ('Tech Val Streams mps chosen',
        {'file': 'valuestreams_chosen.csv',
        'preprocess': [
            {'func': sum_over_cols, 'args': {'group_cols': ['tech', 'year', 'n', 'type'], 'sum_over_cols': ['m']}},
            {'func': scale_column, 'args': {'scale_factor': 1000*CRF_reeds*inflation_mult/1e9, 'column': 'value'}},
        ],
        'presets': collections.OrderedDict((
            ('Bil $ by type over time', {'x':'year','y':'value','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'filter': {}}),
        )),
        }
    ),
    ('Tech Val Streams potential',
        {'file': 'valuestreams_potential.csv',
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': inflation_mult, 'column': 'value_per_unit'}},
        ],
        'presets': collections.OrderedDict((
            ('$/MW by type final', {'x':'var_set','y':'value_per_unit','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'plot_width':'1200', 'filter': {'year':'last','type':{'exclude':['profit','reduced_cost']}}}),
        )),
        }
    ),
    ('Stacked profitability potential',
        {'file': 'valuestreams_potential.csv',
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': inflation_mult, 'column': 'value_per_unit'}},
            {'func': pre_stacked_profitability_potential, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Stacked profitability final', {'x':'var_set','y':'value_per_unit','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'adv_op':'Ratio', 'adv_col':'type', 'adv_col_base':'cost', 'plot_width':'1200', 'filter': {'year':'last','type':{'exclude':['profit','reduced_cost']}}}),
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
