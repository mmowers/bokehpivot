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
import core

this_dir_path = os.path.dirname(os.path.realpath(__file__))
CRF_reeds = 0.077
df_deflator = pd.read_csv(this_dir_path + '/in/inflation.csv', index_col=0)
ILR_UPV = 1.3
ILR_distPV = 1.1

raw_costs = ['fix_cost','var_cost','trans_cost','gp','oper_res_cost','other_cost']
costs = ['Fixed Cost','Variable Cost','Trans Cost','Growth Cost','Ancillary Cost','Other Cost']
raw_values = ['load_pca','res_marg','oper_res','rps','cap_fo_po','surplus','other']
values = ['Energy Value','Capacity Value','Ancillary Value','RPS Value','Cap Fo Po','Curtailment','Other Value']
values_decomp = ['block_dist_load','loc_min_dist_load','real_min_loc_load','block_dist_resmarg','loc_min_dist_resmarg','real_min_loc_resmarg','Ancillary Value','RPS Value','Cap Fo Po','Curtailment','Other Value']
values_load = ['block_dist_load','loc_min_dist_load','real_min_loc_load','Curtailment']
values_resmarg = ['block_dist_resmarg','loc_min_dist_resmarg','real_min_loc_resmarg']
values_resmarg_cap = ['block_cap_dist_resmarg','loc_min_dist_resmarg_cap','real_min_loc_resmarg_cap']

#1. Preprocess functions for results_meta
def scale_column(df_in, **kw):
    df = df_in.copy()
    df[kw['column']] = df[kw['column']] * kw['scale_factor']
    return df

def scale_column_filtered(df_in, **kw):
    df = df_in.copy()
    cond = df[kw['by_column']].isin(kw['by_vals'])
    df.loc[cond, kw['change_column']] = df.loc[cond, kw['change_column']] * kw['scale_factor']
    return df

def scale_pv(df_in, **kw):
    df = df_in.copy()
    df = scale_column_filtered(df, by_column='tech', by_vals=['upv', 'dupv'], change_column=kw['change_column'], scale_factor=1/ILR_UPV)
    df = scale_column_filtered(df, by_column='tech', by_vals=['distpv'], change_column=kw['change_column'], scale_factor=1/ILR_distPV)
    return df

def sum_over_cols(df_in, **kw):
    df = df_in.copy()
    df = df.drop(kw['sum_over_cols'], axis='columns')
    df =  df.groupby(kw['group_cols'], sort=False, as_index =False).sum()
    return df

def apply_inflation(df_in, **kw):
    df = df_in.copy()
    df[kw['column']] = inflate_series(df[kw['column']])
    return df

def inflate_series(ser_in):
    return ser_in * 1/df_deflator.loc[int(core.GL['widgets']['var_dollar_year'].value),'Deflator']

def discount_costs(df_in, **kw):
    df = df_in.copy()
    #inner join the cost_cat_type.csv table to get types of costs (Capital, Operation)
    cost_cat_type = pd.read_csv(this_dir_path + '/in/cost_cat_type.csv')
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

def pre_elec_price(df, **kw):
    df = df.pivot_table(index=['n','year'], columns='elem', values='value').reset_index()
    df.drop(['t2','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16'], axis='columns', inplace=True)
    df.columns.name = None
    df.rename(columns={'t1': 'load', 't3': 'Regulated', 't17': 'Competitive'}, inplace=True)
    df = pd.melt(df, id_vars=['n', 'year', 'load'], value_vars=['Competitive', 'Regulated'], var_name='type', value_name= 'Price ($/MWh)')
    df['Price ($/MWh)'] = inflate_series(df['Price ($/MWh)'])
    return df

def pre_elec_price_components(dfs, **kw):
    df_load = dfs['load'][dfs['load']['type'] == 'reqt']
    df_load = df_load.drop('type', 1)
    df_main = dfs['main']
    df_main['value'] = inflate_series(df_main['value'])
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

def pre_tech_val_streams_raw(dfs, **kw):
    df = dfs['valstream']
    df['$/kW'] = inflate_series(df['$/kW'])
    df = add_chosen_available(df, dfs)
    return df

def pre_tech_val_streams(dfs, **kw):
    #Calculate $/MWh and block values
    df_valstream = dfs['valstream']
    df_load = dfs['load']
    df_price_dist = dfs['prices_nat']
    df_price_ba = dfs['prices_ba']
    df_crf = dfs['CRF']

    if kw['cat'] == 'potential':
        valstream_cols = ['year','tech','new_old','n','type','var_set']
        valstream_val = '$/kW'
        df_valstream.rename(columns={'$/kW': valstream_val}, inplace=True)
        load_val = 'MWh/kW'
        df_valstream = scale_pv(df_valstream, change_column=valstream_val)
        df_load = scale_pv(df_load, change_column=load_val)
        dfs['levels_potential'] = scale_pv(dfs['levels_potential'],change_column='MW')
    elif kw['cat'] == 'chosen':
        valstream_cols = ['year','tech','new_old','n','type']
        valstream_val = '$'
        load_val = 'MWh'
        df_valstream = sum_over_cols(df_valstream, sum_over_cols=['m'], group_cols=valstream_cols)
        df_load = sum_over_cols(df_load, sum_over_cols=['m'], group_cols=['year','tech','new_old','n'])
        df_new_cap = dfs['new_cap']
        df_new_cap = scale_pv(df_new_cap, change_column='kW')
        df_new_cap['kW'] = df_new_cap['kW'] * 1000 #original data is in MW
        df_new_cap['new_old'] = 'new'
        df_new_cap['year'] = pd.to_numeric(df_new_cap['year'])

    #Annualize and adjust by inflation
    df_crf = df_crf[df_crf['crftype']=='crf_20'].copy()
    df_crf.drop(['crftype'], axis='columns', inplace=True)
    df_crf['year'] = pd.to_numeric(df_crf['year'])
    df_valstream = pd.merge(left=df_valstream, right=df_crf, on=['year'], how='left', sort=False)
    df_valstream[valstream_val] = inflate_series(df_valstream[valstream_val]) * df_valstream['crf']
    df_valstream.drop(['crf'], axis='columns', inplace=True)

    #Gather national (dist) and ba-level prices (all in $/MWh assuming block generator with full capacity factor and capacity credit). Adjust by inflation
    df_price_dist = df_price_dist[df_price_dist['type'].isin(['load_pca','res_marg'])].copy()
    df_price_ba = df_price_ba[df_price_ba['type'].isin(['load_pca','res_marg'])].copy()
    df_price_dist['year'] = pd.to_numeric(df_price_dist['year'])
    df_price_ba['year'] = pd.to_numeric(df_price_ba['year'])
    df_price_ba['$/MWh'] = inflate_series(df_price_ba['$/MWh'])
    df_price_dist['$/MWh'] = inflate_series(df_price_dist['$/MWh'])

    #Calculate combined load_pca and res_marg prices (comb) and concatenate into df_price_dist and df_price_ba dataframes.
    df_price_dist_comb = sum_over_cols(df_price_dist, sum_over_cols=['type'], group_cols=['year'])
    df_price_ba_comb = sum_over_cols(df_price_ba, sum_over_cols=['type'], group_cols=['n','year'])
    df_price_dist_comb['type'] = 'comb'
    df_price_ba_comb['type'] = 'comb'
    df_price_dist = pd.concat([df_price_dist,df_price_dist_comb], ignore_index=True)
    df_price_ba = pd.concat([df_price_ba,df_price_ba_comb], ignore_index=True)

    #merge df_price into df_load and calculate energy-based block value streams
    df_block_dist = pd.merge(left=df_load, right=df_price_dist, on=['year'], how='left', sort=False)
    df_block_ba = pd.merge(left=df_load, right=df_price_ba, on=['n','year'], how='left', sort=False)
    df_block_dist[valstream_val] = df_block_dist['$/MWh'] * df_block_dist[load_val]
    df_block_ba[valstream_val] = df_block_ba['$/MWh'] * df_block_ba[load_val]
    df_block_dist.drop(['$/MWh',load_val], axis='columns', inplace=True)
    df_block_ba.drop(['$/MWh',load_val], axis='columns', inplace=True)

    #Add annualized $/kW-yr capacity prices and capacity-based block value streams
    df_price_dist['$/kW'] = df_price_dist['$/MWh'] * 8760/1000
    df_price_ba['$/kW'] = df_price_ba['$/MWh'] * 8760/1000
    if kw['cat'] == 'potential':
        cap_cols = [c for c in valstream_cols if c not in ['type']]
        df_vs_red = df_valstream[cap_cols].drop_duplicates()
        df_block_cap_dist = pd.merge(left=df_vs_red, right=df_price_dist, on=['year'], how='left', sort=False)
        df_block_cap_ba = pd.merge(left=df_vs_red, right=df_price_ba, on=['n','year'], how='left', sort=False)
    elif kw['cat'] == 'chosen':
        df_block_cap_dist = pd.merge(left=df_new_cap, right=df_price_dist, on=['year'], how='left', sort=False)
        df_block_cap_ba = pd.merge(left=df_new_cap, right=df_price_ba, on=['n','year'], how='left', sort=False)
        df_block_cap_dist[valstream_val] = df_block_cap_dist['$/kW'] * df_block_cap_dist['kW']
        df_block_cap_ba[valstream_val] = df_block_cap_ba['$/kW'] * df_block_cap_ba['kW']
        df_block_cap_dist.drop(['$/MWh', '$/kW','kW'], axis='columns', inplace=True)
        df_block_cap_ba.drop(['$/MWh', '$/kW','kW'], axis='columns', inplace=True)

    #rename types to differentiate components
    df_block_dist['type'] = df_block_dist['type'].map({'load_pca': 'block_dist_load', 'res_marg': 'block_dist_resmarg', 'comb': 'block_dist_comb'})
    df_block_ba['type'] = df_block_ba['type'].map({'load_pca': 'block_local_load', 'res_marg': 'block_local_resmarg', 'comb': 'block_local_comb'})
    df_block_cap_dist['type'] = df_block_cap_dist['type'].map({'load_pca': 'block_cap_dist_load', 'res_marg': 'block_cap_dist_resmarg', 'comb': 'block_cap_dist_comb'})
    df_block_cap_ba['type'] = df_block_cap_ba['type'].map({'load_pca': 'block_cap_local_load', 'res_marg': 'block_cap_local_resmarg', 'comb': 'block_cap_local_comb'})

    #Calculate additive adjustments between values of real, local block, and distributed block (value factors are multiplicative adjustments)
    #For load_pca df_real_min_loc represents temporal effects, but for res_marg it represents Capacity credit vs capacity factor.
    #res_marg realy should have special treatment because block value streams are based on energy, and some techs may only be providing reserves.
    #We would find spatial value factor from df_price_ba and df_price_dist, then we divide df_valstream by this spatial value factor to get the "quantity" component.
    df_valstream_comb = df_valstream[df_valstream['type'].isin(['load_pca','res_marg'])].copy()
    df_valstream_comb['type'] = 'comb'
    df_valstream_comb = df_valstream_comb.groupby(valstream_cols, sort=False, as_index =False).sum()
    df_valstream.append(df_valstream_comb)
    if kw['decompose'] == True:
        df_valstream_red = df_valstream[df_valstream['type'].isin(['load_pca','res_marg','comb'])].copy()
        df_real_min_loc = df_valstream_red.set_index(valstream_cols).subtract(df_block_ba.set_index(valstream_cols),fill_value=0).reset_index()
        df_loc_min_dist = df_block_ba.set_index(valstream_cols).subtract(df_block_dist.set_index(valstream_cols),fill_value=0).reset_index()
        df_cap_real_min_loc = df_valstream_red.set_index(valstream_cols).subtract(df_block_cap_ba.set_index(valstream_cols),fill_value=0).reset_index()
        df_cap_loc_min_dist = df_block_cap_ba.set_index(valstream_cols).subtract(df_block_cap_dist.set_index(valstream_cols),fill_value=0).reset_index()
        #rename types to differentiate components
        df_cap_real_min_loc['type'] = df_cap_real_min_loc['type'].map({'load_pca': 'real_min_loc_load_cap', 'res_marg': 'real_min_loc_resmarg_cap', 'comb': 'real_min_loc_comb_cap'})
        df_cap_loc_min_dist['type'] = df_cap_loc_min_dist['type'].map({'load_pca': 'loc_min_dist_load_cap', 'res_marg': 'loc_min_dist_resmarg_cap', 'comb': 'loc_min_dist_comb_cap'})
        df_real_min_loc['type'] = df_real_min_loc['type'].map({'load_pca': 'real_min_loc_load', 'res_marg': 'real_min_loc_resmarg', 'comb': 'real_min_loc_comb'})
        df_loc_min_dist['type'] = df_loc_min_dist['type'].map({'load_pca': 'loc_min_dist_load', 'res_marg': 'loc_min_dist_resmarg', 'comb': 'loc_min_dist_comb'})

    #Reformat Energy Output
    df_load['type'] = load_val
    df_load.rename(columns={load_val: valstream_val}, inplace=True) #rename just so we can concatenate, even though units are load_val

    #Add Total Cost (positive)
    df_cost = df_valstream[df_valstream['type'].isin(raw_costs)].copy()
    df_cost['type'] = 'total cost'
    df_cost = df_cost.groupby(valstream_cols, sort=False, as_index =False).sum()
    df_cost[valstream_val] = df_cost[valstream_val]*-1

    #Add Total Value
    df_val = df_valstream[df_valstream['type'].isin(raw_values)].copy()
    df_val['type'] = 'total value'
    df_val = df_val.groupby(valstream_cols, sort=False, as_index =False).sum()

    #Add System LCOE numerator (positive), which is national distributed price times total cost (denominator is total value)
    df_slcoenum = pd.merge(left=df_cost, right=df_price_dist_comb[['year','$/MWh']], on=['year'], how='left', sort=False)
    df_slcoenum[valstream_val] = df_slcoenum[valstream_val] * df_slcoenum['$/MWh']
    df_slcoenum['type'] = 'sys_lcoe_num'
    df_slcoenum.drop(['$/MWh'], axis='columns', inplace=True)

    #Add OLD System LCOE base value for base price calculation (negative)
    df_baseval = df_block_dist[df_block_dist['type'] == 'block_dist_comb'].copy()
    df_baseval['type'] = 'base_value'
    df_baseval[valstream_val] = df_baseval[valstream_val]*-1

    #Combine dataframes
    df_list = [df_valstream, df_load, df_cost, df_val, df_slcoenum, df_block_ba, df_block_dist, df_block_cap_ba, df_block_cap_dist, df_baseval]
    if kw['decompose'] == True:
        df_list = df_list + [df_real_min_loc,df_loc_min_dist,df_cap_real_min_loc,df_cap_loc_min_dist]
    if kw['cat'] == 'chosen':
        #Reformat Capacity Output
        df_new_cap['type'] = 'kW'
        df_new_cap.rename(columns={'kW': valstream_val}, inplace=True) #rename just so we can concatenate, even though units are different
        df_list.append(df_new_cap)
        df = pd.concat(df_list, ignore_index=True)
    elif kw['cat'] == 'potential':
        df = pd.concat(df_list, ignore_index=True)
        df = add_chosen_available(df, dfs)
    df.rename(columns={'type': 'cost_val_type'}, inplace=True)
    return df

def add_chosen_available(df, dfs):
    #Add chosen column to indicate if this resource was built.
    df = pd.merge(left=df, right=dfs['levels_potential'], on=['year','tech','new_old','var_set'], how='left', sort=False)
    df.rename(columns={'MW': 'chosen'}, inplace=True)
    df['chosen'] = df['chosen'].fillna(value='no')
    df.loc[df['chosen'] != 'no', 'chosen'] = "yes"
    #Add available column to indicate if the resource was available to be built.
    df_avail = dfs['available_potential']
    df_avail['available'] = 'yes'
    df = pd.merge(left=df, right=df_avail, on=['year','var_set'], how='left', sort=False)
    df.loc[~df['tech'].isin(['wind-ons','wind-ofs','upv','dupv']), 'available'] = 'yes'
    df['available'] = df['available'].fillna(value='no')
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
    'cost_val_type':{
        'type': 'string',
        'map': this_dir_path + '/in/cost_val_map.csv',
        'style': this_dir_path + '/in/cost_val_style.csv',
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
            {'func': scale_pv, 'args': {'change_column': 'Capacity (GW)'}},
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
            {'func': scale_pv, 'args': {'change_column': 'Capacity (GW)'}},
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
            {'func': scale_pv, 'args': {'change_column': 'Capacity (GW)'}},
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
    ('Sys Cost (Bil $)',
        {'file': 'systemcost.gdx',
        'param': 'aSystemCost_ba',
        'columns': ['cost_cat', 'n', 'year', 'Cost (Bil $)'],
        'index': ['cost_cat', 'n', 'year'],
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
    ('Gen by m national (GW)',
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
    ('Gen by m regional (GW)',
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
    ('Elec Price ($/MWh)',
        {'file': 'Reporting.gdx',
        'param': 'ElecPriceOut',
        'columns': ['n', 'year', 'elem', 'value'],
        'index': ['n', 'year', 'type'],
        'preprocess': [
            {'func': pre_elec_price, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('National',{'x':'year','y':'Price ($/MWh)', 'y_agg':'Weighted Ave', 'y_weight':'load', 'series':'scenario', 'explode': 'type', 'chart_type':'Line'}),
            ('National Scenario',{'x':'year','y':'Price ($/MWh)', 'y_agg':'Weighted Ave', 'y_weight':'load', 'series':'type', 'explode': 'scenario', 'chart_type':'Line'}),
            ('Census',{'x':'year','y':'Price ($/MWh)', 'y_agg':'Weighted Ave', 'y_weight':'load', 'series':'scenario', 'explode': 'censusregions', 'explode_group': 'type', 'chart_type':'Line'}),
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
    ('Tech Val Streams chosen raw',
        {'file': 'valuestreams/valuestreams_chosen.csv',
        'preprocess': [
            {'func': sum_over_cols, 'args': {'group_cols': ['tech', 'new_old', 'year', 'n', 'type'], 'sum_over_cols': ['m']}},
            {'func': apply_inflation, 'args': {'column': '$'}},
            {'func': scale_column, 'args': {'scale_factor': CRF_reeds/1e9, 'column': '$'}},
        ],
        'presets': collections.OrderedDict((
            ('New Bil $ by type over time', {'x':'year','y':'$','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'new_old':['new']}}),
            ('Old Bil $ by type over time', {'x':'year','y':'$','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'new_old':['old']}}),
            ('Mixed Bil $ by type over time', {'x':'year','y':'$','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'new_old':['mixed']}}),
            ('Retire Bil $ by type over time', {'x':'year','y':'$','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'new_old':['retire']}}),
            ('New Bil $ Cost by tech over time', {'x':'year','y':'$','series':'tech', 'explode': 'scenario', 'chart_type':'Bar', 'bar_width':'1.75', 'y_scale':'-1', 'filter': {'new_old':['new'], 'type':['fix_cost','gp','trans_cost','var_cost']}}),
            ('New Bil $ by type over time agg', {'x':'year','y':'$','series':'type', 'explode': 'scenario', 'chart_type':'Bar', 'bar_width':'1.75', 'filter': {'new_old':['new']}}),
        )),
        }
    ),
    ('Tech Val Streams chosen',
        {'sources': [
            {'name': 'valstream', 'file': 'valuestreams/valuestreams_chosen.csv'},
            {'name': 'load', 'file': 'valuestreams/load_pca_chosen.csv'},
            {'name': 'prices_nat', 'file': 'MarginalPrices.gdx', 'param': 'p_block_nat_ann', 'columns': ['type','year','$/MWh']},
            {'name': 'prices_ba', 'file': 'MarginalPrices.gdx', 'param': 'p_block_ba_ann', 'columns': ['n','type','year','$/MWh']},
            {'name': 'new_cap', 'file': 'CONVqn.gdx', 'param': 'CONVqn_newallyears', 'columns': ['tech', 'n', 'year', 'kW']},
            {'name': 'CRF', 'file': '../input-data.gdx', 'param': 'CRF_allyears', 'columns': ['crftype','year','crf']},
        ],
        'preprocess': [
            {'func': pre_tech_val_streams, 'args': {'cat':'chosen','decompose':False}},
        ],
        'presets': collections.OrderedDict((
            ('$/kW by type over time', {'x':'year','y':'$','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'kW', 'chart_type':'Bar', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'cost_val_type':costs+values+['kW'],'new_old':['new']}}),
            ('$/kW by type final', {'chart_type':'Bar', 'x':'tech', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'kW', 'sync_axes':'No', 'bar_width':r'.9', 'bar_sort': 'Descending', 'plot_width':'600', 'plot_height':'600', 'filter': {'new_old':['new'], 'tech':{'exclude':['Distributed PV (AC)','distpv']}, 'cost_val_type':costs+values+['kW'], 'year':'last', }}),
            ('Resmarg $/kW decomposed over time', {'x':'year','y':'$','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'kW', 'chart_type':'Bar', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'cost_val_type':values_resmarg_cap+['kW'],'new_old':['new']}}),
            ('Resmarg $/kW decomposed final', {'chart_type':'Bar', 'x':'tech', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'kW', 'sync_axes':'No', 'bar_width':r'.9', 'bar_sort': 'Descending', 'plot_width':'600', 'plot_height':'600', 'filter': {'new_old':['new'], 'tech':{'exclude':['Distributed PV (AC)','distpv']}, 'cost_val_type':values_resmarg_cap+['kW'], 'year':'last', }}),

            ('$/MWh by type over time', {'x':'year','y':'$','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'MWh', 'chart_type':'Bar', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'cost_val_type':costs+values+['MWh'],'new_old':['new']}}),
            ('$/MWh by type decomposed over time', {'x':'year','y':'$','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'MWh', 'chart_type':'Bar', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'cost_val_type':costs+values_decomp+['MWh'],'new_old':['new']}}),
            ('$/MWh by type final', {'chart_type':'Bar', 'x':'tech', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'MWh', 'sync_axes':'No', 'bar_width':r'.9', 'bar_sort': 'Descending', 'plot_width':'600', 'plot_height':'600', 'filter': {'new_old':['new'], 'tech':{'exclude':['Distributed PV (AC)','distpv']}, 'cost_val_type':costs+values+['MWh'], 'year':'last', }}),
            ('$/MWh by type decomposed final', {'chart_type':'Bar', 'x':'tech', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'MWh', 'sync_axes':'No', 'bar_width':r'.9', 'bar_sort': 'Descending', 'plot_width':'600', 'plot_height':'600', 'filter': {'new_old':['new'], 'tech':{'exclude':['Distributed PV (AC)','distpv']}, 'cost_val_type':costs+values_decomp+['MWh'], 'year':'last', }}),
            ('LCOE over time', {'chart_type':'Bar', 'x':'year', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'explode_group':'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'MWh', 'sync_axes':'No', 'bar_width':r'1.75', 'y_scale':'-1', 'filter': {'new_old':['new'], 'cost_val_type':costs + ['MWh'], }}),
            ('LCOE final', {'chart_type':'Bar', 'x':'tech', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'MWh', 'sync_axes':'No', 'bar_width':r'.9', 'bar_sort': 'Ascending', 'plot_width':'600', 'plot_height':'600', 'y_scale':'-1', 'filter': {'new_old':['new'], 'tech':{'exclude':['Distributed PV (AC)','distpv']}, 'cost_val_type':costs+['MWh'], 'year':'last', }}),
            ('Load $/MWh decomposed over time', {'x':'year','y':'$','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'MWh', 'chart_type':'Bar', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'cost_val_type':values_load+['MWh'],'new_old':['new']}}),
            ('Load $/MWh decomposed final', {'chart_type':'Bar', 'x':'tech', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'MWh', 'sync_axes':'No', 'bar_width':r'.9', 'bar_sort': 'Descending', 'plot_width':'600', 'plot_height':'600', 'filter': {'new_old':['new'], 'tech':{'exclude':['Distributed PV (AC)','distpv']}, 'cost_val_type':values_load+['MWh'], 'year':'last', }}),
            ('Resmarg $/MWh decomposed over time', {'x':'year','y':'$','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'MWh', 'chart_type':'Bar', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'cost_val_type':values_resmarg+['MWh'],'new_old':['new']}}),
            ('Resmarg $/MWh decomposed final', {'chart_type':'Bar', 'x':'tech', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'MWh', 'sync_axes':'No', 'bar_width':r'.9', 'bar_sort': 'Descending', 'plot_width':'600', 'plot_height':'600', 'filter': {'new_old':['new'], 'tech':{'exclude':['Distributed PV (AC)','distpv']}, 'cost_val_type':values_resmarg+['MWh'], 'year':'last', }}),

            ('Profitability over time', {'x':'year','y':'$','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'total cost', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'cost_val_type':costs+values+['total cost'],'new_old':['new']}}),
            ('Profitability decomposed over time', {'x':'year','y':'$','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'total cost', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'cost_val_type':costs+values_decomp+['total cost'],'new_old':['new']}}),
            ('Profitability final', {'chart_type':'Bar', 'x':'tech', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'total cost', 'sync_axes':'No', 'bar_width':r'.9', 'bar_sort': 'Descending', 'plot_width':'600', 'plot_height':'600', 'filter': {'new_old':['new'], 'tech':{'exclude':['Distributed PV (AC)','distpv']}, 'cost_val_type':costs+values+['total cost'], 'year':'last', }}),
            ('Profitability decomposed final', {'chart_type':'Bar', 'x':'tech', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'total cost', 'sync_axes':'No', 'bar_width':r'.9', 'bar_sort': 'Descending', 'plot_width':'600', 'plot_height':'600', 'filter': {'new_old':['new'], 'tech':{'exclude':['Distributed PV (AC)','distpv']}, 'cost_val_type':costs+values_decomp+['total cost'], 'year':'last', }}),
            ('Load profit decomposed over time', {'x':'year','y':'$','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'total cost', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'cost_val_type':values_load+['total cost'],'new_old':['new']}}),
            ('Load profit decomposed final', {'chart_type':'Bar', 'x':'tech', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'total cost', 'sync_axes':'No', 'bar_width':r'.9', 'bar_sort': 'Descending', 'plot_width':'600', 'plot_height':'600', 'filter': {'new_old':['new'], 'tech':{'exclude':['Distributed PV (AC)','distpv']}, 'cost_val_type':values_load+['total cost'], 'year':'last', }}),
            ('Resmarg profit decomposed over time', {'x':'year','y':'$','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'total cost', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'cost_val_type':values_resmarg+['total cost'],'new_old':['new']}}),
            ('Resmarg profit decomposed final', {'chart_type':'Bar', 'x':'tech', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'total cost', 'sync_axes':'No', 'bar_width':r'.9', 'bar_sort': 'Descending', 'plot_width':'600', 'plot_height':'600', 'filter': {'new_old':['new'], 'tech':{'exclude':['Distributed PV (AC)','distpv']}, 'cost_val_type':values_resmarg+['total cost'], 'year':'last', }}),
            ('Resmarg cap profit decomposed over time', {'x':'year','y':'$','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'total cost', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'cost_val_type':values_resmarg_cap+['total cost'],'new_old':['new']}}),
            ('Resmarg cap profit decomposed final', {'chart_type':'Bar', 'x':'tech', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'total cost', 'sync_axes':'No', 'bar_width':r'.9', 'bar_sort': 'Descending', 'plot_width':'600', 'plot_height':'600', 'filter': {'new_old':['new'], 'tech':{'exclude':['Distributed PV (AC)','distpv']}, 'cost_val_type':values_resmarg_cap+['total cost'], 'year':'last', }}),

            ('System LCOE over time', {'chart_type':'Bar', 'x':'year','y':'$','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'total value', 'sync_axes':'No', 'filter': {'cost_val_type':['sys_lcoe_num','total value'],'new_old':['new']}}),
            ('System LCOE final', {'chart_type':'Bar', 'x':'tech','y':'$','series':'cost_val_type', 'explode': 'scenario', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'total value', 'sync_axes':'No', 'bar_width':r'.9', 'bar_sort': 'Ascending', 'filter': {'cost_val_type':['sys_lcoe_num','total value'],'new_old':['new'],'year':'last','exclude':['Distributed PV (AC)','distpv']}}),
            ('OLD System LCOE over time', {'chart_type':'Bar', 'x':'year','y':'$','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'MWh', 'y_scale':'-1', 'sync_axes':'No', 'filter': {'cost_val_type':costs+values+['MWh','Base Price'],'new_old':['new']}}),
            ('OLD System LCOE final', {'chart_type':'Bar', 'x':'tech','y':'$','series':'cost_val_type', 'explode': 'scenario', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'MWh', 'y_scale':'-1', 'sync_axes':'No', 'bar_width':r'.9', 'bar_sort': 'Ascending', 'filter': {'cost_val_type':costs+values+['MWh','Base Price'],'new_old':['new'],'year':'last','exclude':['Distributed PV (AC)','distpv']}}),

            ('Value factor Combined Dist by type final', {'x':'n','y':'$','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_dist_comb', 'chart_type':'Bar', 'plot_width':'600', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'year':'last','cost_val_type':['block_dist_comb']+values,'new_old':['new']}}),
            ('Value factor Combined Local by type final', {'x':'n','y':'$','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_local_comb', 'chart_type':'Bar', 'plot_width':'600', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'year':'last','cost_val_type':['block_local_comb']+values,'new_old':['new']}}),
            ('Value factor Combined over time', {'chart_type':'Bar', 'x':'year', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'explode_group':'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_dist_comb', 'sync_axes':'No', 'bar_width':r'1.75', 'filter': {'new_old':['new'], 'cost_val_type':['block_dist_comb']+values, }}),
            ('Value factor Combined Temporal over time', {'chart_type':'Bar', 'x':'year', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'explode_group':'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_local_comb', 'sync_axes':'No', 'bar_width':r'1.75', 'filter': {'new_old':['new'], 'cost_val_type':['block_local_comb']+values, }}),
            ('Value factor Combined Spatial over time', {'chart_type':'Bar', 'x':'year', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'explode_group':'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_dist_comb', 'sync_axes':'No', 'bar_width':r'1.75', 'filter': {'new_old':['new'], 'cost_val_type':['block_dist_comb','block_local_comb'], }}),

            ('Value factor Load Dist by type final', {'x':'n','y':'$','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_dist_load', 'chart_type':'Bar', 'plot_width':'600', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'year':'last','cost_val_type':['block_dist_load','Energy Value','Curtailment'],'new_old':['new']}}),
            ('Value factor Load Local by type final', {'x':'n','y':'$','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_local_load', 'chart_type':'Bar', 'plot_width':'600', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'year':'last','cost_val_type':['block_local_load','Energy Value','Curtailment'],'new_old':['new']}}),
            ('Value factor Load over time', {'chart_type':'Bar', 'x':'year', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'explode_group':'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_dist_load', 'sync_axes':'No', 'bar_width':r'1.75', 'filter': {'new_old':['new'], 'cost_val_type':['block_dist_load','Energy Value','Curtailment'], }}),
            ('Value factor Load Temporal over time', {'chart_type':'Line', 'x':'year', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'explode_group':'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_local_load', 'sync_axes':'No', 'bar_width':r'1.75', 'filter': {'new_old':['new'], 'cost_val_type':['block_local_load','Energy Value'], }}),
            ('Value factor Load Spatial over time', {'chart_type':'Line', 'x':'year', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'explode_group':'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_dist_load', 'sync_axes':'No', 'bar_width':r'1.75', 'filter': {'new_old':['new'], 'cost_val_type':['block_dist_load','block_local_load'], }}),
            ('Curtail Frac over time', {'chart_type':'Line', 'x':'year', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'explode_group':'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'Energy Value', 'y_scale':'-1', 'sync_axes':'No', 'bar_width':r'1.75', 'filter': {'new_old':['new'], 'cost_val_type':['Energy Value','Curtailment'], }}),

            ('Value factor Res Marg Dist by type final', {'x':'n','y':'$','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_dist_resmarg', 'chart_type':'Bar', 'plot_width':'600', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'year':'last','cost_val_type':['block_dist_resmarg','Capacity Value'],'new_old':['new']}}),
            ('Value factor Res Marg Local by type final', {'x':'n','y':'$','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_local_resmarg', 'chart_type':'Bar', 'plot_width':'600', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'year':'last','cost_val_type':['block_local_resmarg','Capacity Value'],'new_old':['new']}}),
            ('Value factor Res Marg over time', {'chart_type':'Line', 'x':'year', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'explode_group':'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_dist_resmarg', 'sync_axes':'No', 'bar_width':r'1.75', 'filter': {'new_old':['new'], 'cost_val_type':['block_dist_resmarg','Capacity Value'], }}),
            ('Value factor Res Marg CV/CF over time', {'chart_type':'Line', 'x':'year', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'explode_group':'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_local_resmarg', 'sync_axes':'No', 'bar_width':r'1.75', 'filter': {'new_old':['new'], 'cost_val_type':['block_local_resmarg','Capacity Value'], }}),
            ('Value factor Res Marg Spatial over time', {'chart_type':'Line', 'x':'year', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'explode_group':'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_dist_resmarg', 'sync_axes':'No', 'bar_width':r'1.75', 'filter': {'new_old':['new'], 'cost_val_type':['block_dist_resmarg','block_local_resmarg'], }}),

            ('CC Res Marg by type final', {'x':'n','y':'$','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_cap_local_resmarg', 'chart_type':'Bar', 'plot_width':'600', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'year':'last','cost_val_type':['block_cap_local_resmarg','Capacity Value'],'new_old':['new']}}),
            ('CC Res Marg over time', {'chart_type':'Line', 'x':'year', 'y':'$', 'series':'cost_val_type', 'explode':'scenario', 'explode_group':'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_cap_local_resmarg', 'sync_axes':'No', 'bar_width':r'1.75', 'filter': {'new_old':['new'], 'cost_val_type':['block_cap_local_resmarg','Capacity Value'], }}),
        )),
        }
    ),
    ('Tech Val Streams potential raw',
        {'sources': [
            {'name': 'valstream', 'file': 'valuestreams/valuestreams_potential.csv'},
            {'name': 'levels_potential', 'file': 'valuestreams/levels_potential.csv'},
            {'name': 'available_potential', 'file': 'valuestreams/available_potential.csv'},
        ],
        'preprocess': [
            {'func': pre_tech_val_streams_raw, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('$/kW by type final', {'x':'var_set','y':'$/kW','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'year':'last','type':{'exclude':['profit','reduced_cost']},'new_old':['new']}}),
            ('$/kW by type final p60', {'x':'var_set','y':'$/kW','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'n':['p60'],'year':'last','type':{'exclude':['profit','reduced_cost']},'new_old':['new']}}),
            ('$/kW by type retire final', {'x':'var_set','y':'$/kW','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'year':'last','type':{'exclude':['profit','reduced_cost']},'new_old':['retire']}}),
            ('$/kW by type retire final p60', {'x':'var_set','y':'$/kW','series':'type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'n':['p60'],'year':'last','type':{'exclude':['profit','reduced_cost']},'new_old':['retire']}}),
        )),
        }
    ),
    ('Tech Val Streams potential',
        {'sources': [
            {'name': 'valstream', 'file': 'valuestreams/valuestreams_potential.csv'},
            {'name': 'load', 'file': 'valuestreams/load_pca_potential.csv'},
            {'name': 'levels_potential', 'file': 'valuestreams/levels_potential.csv'},
            {'name': 'available_potential', 'file': 'valuestreams/available_potential.csv'},
            {'name': 'prices_nat', 'file': 'MarginalPrices.gdx', 'param': 'p_block_nat_ann', 'columns': ['type','year','$/MWh']},
            {'name': 'prices_ba', 'file': 'MarginalPrices.gdx', 'param': 'p_block_ba_ann', 'columns': ['n','type','year','$/MWh']},
            {'name': 'CRF', 'file': '../input-data.gdx', 'param': 'CRF_allyears', 'columns': ['crftype','year','crf']},
        ],
        'preprocess': [
            {'func': pre_tech_val_streams, 'args': {'cat':'potential','decompose':False}},
        ],
        'presets': collections.OrderedDict((
            ('$/kW by type final', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'year':'last','cost_val_type':costs+values,'new_old':['new']}}),
            ('$/kW by type final decomposed', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'year':'last','cost_val_type':costs+values_decomp,'new_old':['new']}}),
            ('$/kW by type final p60', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'n':['p60'],'year':'last','cost_val_type':costs+values,'new_old':['new']}}),
            ('$/kW by type final p60 decomposed', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'n':['p60'],'year':'last','cost_val_type':costs+values_decomp,'new_old':['new']}}),
            ('Load $/kW by type final decomposed', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'year':'last','cost_val_type':values_load,'new_old':['new']}}),
            ('Load $/kW by type final p60 decomposed', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'n':['p60'],'year':'last','cost_val_type':values_load,'new_old':['new']}}),
            ('Resmarg $/kW by type final decomposed', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'year':'last','cost_val_type':values_resmarg,'new_old':['new']}}),
            ('Resmarg $/kW by type final p60 decomposed', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'n':['p60'],'year':'last','cost_val_type':values_resmarg,'new_old':['new']}}),

            ('$/MWh by type final', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'MWh/kW', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'year':'last','cost_val_type':costs+values+['MWh/kW'],'new_old':['new']}}),
            ('$/MWh by type final p60', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'MWh/kW', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'n':['p60'],'year':'last','cost_val_type':costs+values+['MWh/kW'],'new_old':['new']}}),
            ('$/MWh by type final decomposed', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'MWh/kW', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'year':'last','cost_val_type':costs+values_decomp+['MWh/kW'],'new_old':['new']}}),
            ('$/MWh by type final decomposed p60', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'MWh/kW', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'n':['p60'],'year':'last','cost_val_type':costs+values_decomp+['MWh/kW'],'new_old':['new']}}),
            ('LCOE final', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'MWh/kW', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Ascending', 'sync_axes':'No', 'y_scale':'-1', 'filter': {'year':'last','cost_val_type':costs+['MWh/kW'],'new_old':['new']}}),
            ('LCOE final p60', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'MWh/kW', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Ascending', 'sync_axes':'No', 'y_scale':'-1', 'filter': {'n':['p60'],'year':'last','cost_val_type':costs+['MWh/kW'],'new_old':['new']}}),
            ('Load $/MWh by type final decomposed', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'MWh/kW', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'year':'last','cost_val_type':values_load+['MWh/kW'],'new_old':['new']}}),
            ('Load $/MWh by type final decomposed p60', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'MWh/kW', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'n':['p60'],'year':'last','cost_val_type':values_load+['MWh/kW'],'new_old':['new']}}),
            ('Resmarg $/MWh by type final decomposed', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'MWh/kW', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'year':'last','cost_val_type':values_resmarg+['MWh/kW'],'new_old':['new']}}),
            ('Resmarg $/MWh by type final decomposed p60', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'MWh/kW', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'n':['p60'],'year':'last','cost_val_type':values_resmarg+['MWh/kW'],'new_old':['new']}}),

            ('Stacked profitability final', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'total cost', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'year':'last','cost_val_type':costs+values+['total cost'],'new_old':['new']}}),
            ('Stacked profitability final p60', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'total cost', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'n':['p60'],'year':'last','cost_val_type':costs+values+['total cost'],'new_old':['new']}}),
            ('Stacked profitability final decomposed', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'total cost', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'year':'last','cost_val_type':costs+values_decomp+['total cost'],'new_old':['new']}}),
            ('Stacked profitability final decomposed p60', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'total cost', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'n':['p60'],'year':'last','cost_val_type':costs+values_decomp+['total cost'],'new_old':['new']}}),
            ('Load profit final decomposed', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'total cost', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'year':'last','cost_val_type':values_load+['total cost'],'new_old':['new']}}),
            ('Load profit final decomposed p60', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'total cost', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'n':['p60'],'year':'last','cost_val_type':values_load+['total cost'],'new_old':['new']}}),
            ('Resmarg profit final decomposed', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'total cost', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'year':'last','cost_val_type':values_resmarg+['total cost'],'new_old':['new']}}),
            ('Resmarg profit final decomposed p60', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'chart_type':'Bar', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'total cost', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'n':['p60'],'year':'last','cost_val_type':values_resmarg+['total cost'],'new_old':['new']}}),

            ('System LCOE final p60', {'chart_type':'Bar', 'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'total value', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Ascending', 'sync_axes':'No', 'filter': {'n':['p60'],'year':'last','cost_val_type':['total value','sys_lcoe_num'],'new_old':['new']}}),
            ('OLD System LCOE final p60', {'chart_type':'Bar', 'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'MWh/kW', 'y_scale':'-1', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Ascending', 'sync_axes':'No', 'filter': {'n':['p60'],'year':'last','cost_val_type':costs+values+['MWh/kW','Base Price'],'new_old':['new']}}),

            ('Value factor Combined Dist by type final', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_dist_comb', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'year':'last','cost_val_type':['block_dist_comb']+values,'new_old':['new']}}),
            ('Value factor Combined Dist by type final p60', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_dist_comb', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'n':['p60'],'year':'last','cost_val_type':['block_dist_comb']+values,'new_old':['new']}}),
            ('Value factor Combined Local by type final', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_local_comb', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'year':'last','cost_val_type':['block_local_comb']+values,'new_old':['new']}}),
            ('Value factor Combined Local by type final p60', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_local_comb', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'n':['p60'],'year':'last','cost_val_type':['block_local_comb']+values,'new_old':['new']}}),

            ('Value factor Load Dist by type final', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_dist_load', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'year':'last','cost_val_type':['block_dist_load','Energy Value','Curtailment'],'new_old':['new']}}),
            ('Value factor Load Dist by type final p60', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_dist_load', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'n':['p60'],'year':'last','cost_val_type':['block_dist_load','Energy Value','Curtailment'],'new_old':['new']}}),
            ('Value factor Load Local by type final', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_local_load', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'year':'last','cost_val_type':['block_local_load','Energy Value','Curtailment'],'new_old':['new']}}),
            ('Value factor Load Local by type final p60', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_local_load', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'n':['p60'],'year':'last','cost_val_type':['block_local_load','Energy Value','Curtailment'],'new_old':['new']}}),

            ('Value factor Res Marg Dist by type final', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_dist_resmarg', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'year':'last','cost_val_type':['block_dist_resmarg','Capacity Value'],'new_old':['new']}}),
            ('Value factor Res Marg Dist by type final p60', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_dist_resmarg', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'n':['p60'],'year':'last','cost_val_type':['block_dist_resmarg','Capacity Value'],'new_old':['new']}}),
            ('Value factor Res Marg Local by type final', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_local_resmarg', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'year':'last','cost_val_type':['block_local_resmarg','Capacity Value'],'new_old':['new']}}),
            ('Value factor Res Marg Local by type final p60', {'x':'var_set','y':'$/kW','series':'cost_val_type', 'explode': 'scenario', 'explode_group': 'tech', 'adv_op':'Ratio', 'adv_col':'cost_val_type', 'adv_col_base':'block_local_resmarg', 'chart_type':'Bar', 'plot_width':'1200', 'bar_width':'0.9', 'bar_sort': 'Descending', 'sync_axes':'No', 'filter': {'n':['p60'],'year':'last','cost_val_type':['block_local_resmarg','Capacity Value'],'new_old':['new']}}),
        )),
        }
    ),
    ('Marginal Prices',
        {'file': 'MarginalPrices.gdx',
        'param': 'pmarg_BA_allyrs',
        'columns': ['n', 'm', 'type', 'year', 'value'],
        }
    ),
    ('BA Ann Marginal Prices',
        {'file': 'MarginalPrices.gdx',
        'param': 'pmarg_BA_ann_allyrs',
        'columns': ['n', 'type', 'year', '$/MWh'],
        'preprocess': [
            {'func': apply_inflation, 'args': {'column': '$/MWh'}},
        ],
        'presets': collections.OrderedDict((
            ('Final load and res_marg annual ba prices', {'x':'n', 'y':'$/MWh', 'explode':'type', 'explode_group':'scenario', 'plot_width':r'1200', 'filter': {'type':['load_pca','res_marg'], 'year':'last'}}),
        )),
        }
    ),
    ('State Ann Marginal Prices',
        {'file': 'MarginalPrices.gdx',
        'param': 'pmarg_st_ann_allyrs',
        'columns': ['st','type', 'year', '$/MWh'],
        'preprocess': [
            {'func': apply_inflation, 'args': {'column': '$/MWh'}},
        ],
        'presets': collections.OrderedDict((
            ('Final load and res_marg annual st prices', {'x':'st', 'y':'$/MWh', 'explode':'type', 'explode_group':'scenario', 'plot_width':r'1200', 'filter': {'type':['load_pca','res_marg'], 'year':'last'}}),
        )),
        }
    ),
    ('Nat Ann Marginal Prices',
        {'file': 'MarginalPrices.gdx',
        'param': 'pmarg_nat_ann_allyrs',
        'columns': ['type', 'year', '$/MWh'],
        'preprocess': [
            {'func': apply_inflation, 'args': {'column': '$/MWh'}},
        ],
        'presets': collections.OrderedDict((
            ('Major Prices over time', {'chart_type':'Line', 'x':'year', 'y':'$/MWh', 'series':'type', 'explode':'scenario', 'filter': {'type':['load_pca','oper_res_reqt-flex','oper_res_reqt-reg','oper_res_reqt-spin','res_marg'], }}),
        )),
        }
    ),
    ('BA Ann Marginal Block Prices',
        {'file': 'MarginalPrices.gdx',
        'param': 'p_block_ba_ann',
        'columns': ['n', 'type', 'year', '$/MWh'],
        'preprocess': [
            {'func': apply_inflation, 'args': {'column': '$/MWh'}},
        ],
        'presets': collections.OrderedDict((
            ('Final load and res_marg annual ba prices', {'x':'n', 'y':'$/MWh', 'explode':'type', 'explode_group':'scenario', 'plot_width':r'1200', 'filter': {'type':['load_pca','res_marg'], 'year':'last'}}),
        )),
        }
    ),
    ('Nat Ann Marginal Block Prices',
        {'file': 'MarginalPrices.gdx',
        'param': 'p_block_nat_ann',
        'columns': ['type', 'year', '$/MWh'],
        'preprocess': [
            {'func': apply_inflation, 'args': {'column': '$/MWh'}},
        ],
        'presets': collections.OrderedDict((
            ('Major Prices over time', {'chart_type':'Line', 'x':'year', 'y':'$/MWh', 'series':'type', 'explode':'scenario', 'filter': {'type':['load_pca','res_marg'], }}),
        )),
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
    ('JEDI Outputs csv',
        {'file': "JEDI_out.csv",
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
        'columns': ['cost_cat', 'year', 'Cost ($)'],
        'preprocess': [
            {'func': apply_inflation, 'args': {'column': 'Cost ($)'}},
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
