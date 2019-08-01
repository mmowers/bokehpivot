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

rb_globs = {'output_subdir':'\\outputs\\', 'test_file':'cap.csv', 'report_subdir':'/reeds2'}
this_dir_path = os.path.dirname(os.path.realpath(__file__))
CRF_reeds = 0.077
df_deflator = pd.read_csv(this_dir_path + '/in/inflation.csv', index_col=0)
costs_orig_inv = ['Capital no policy']
costs_pol_inv = ['Capital with policy','PTC']
coststreams = ['_obj','eq_bioused','eq_gasused']

#1. Preprocess functions for results_meta
def scale_column(df, **kw):
    df[kw['column']] = df[kw['column']] * kw['scale_factor']
    return df

def scale_column_filtered(df, **kw):
    cond = df[kw['by_column']].isin(kw['by_vals'])
    df.loc[cond, kw['change_column']] = df.loc[cond, kw['change_column']] * kw['scale_factor']
    return df

def sum_over_cols(df, **kw):
    df = df.drop(kw['drop_cols'], axis='columns')
    df =  df.groupby(kw['group_cols'], sort=False, as_index =False).sum()
    return df

def apply_inflation(df, **kw):
    df[kw['column']] = inflate_series(df[kw['column']])
    return df

def inflate_series(ser_in):
    return ser_in * 1/df_deflator.loc[int(core.GL['widgets']['var_dollar_year'].value), 'Deflator']

def pre_systemcost(dfs, **kw):
    df = dfs['sc']
    #apply inflation and adjust to billion dollars
    df['Cost (Bil $)'] = inflate_series(df['Cost (Bil $)']) * 1e-9
    d = float(core.GL['widgets']['var_discount_rate'].value)
    y0 = int(core.GL['widgets']['var_pv_year'].value)
    #Annualize if specified
    if 'annualize' in kw:
        cost_cats_df = df['cost_cat'].unique().tolist()
        #Gather lists of capital and operation labels
        df_cost_type = pd.read_csv(this_dir_path + '/in/reeds2/cost_cat_type.csv')
        #Make sure all cost categories in df are in df_cost_type and throw error if not!!
        if not set(cost_cats_df).issubset(df_cost_type['cost_cat'].values.tolist()):
            print('WARNING: Not all cost categories have been mapped!!!')
        cap_type_ls = [c for c in cost_cats_df if c in df_cost_type[df_cost_type['type']=='Capital']['cost_cat'].tolist()]
        op_type_ls = [c for c in cost_cats_df if c in df_cost_type[df_cost_type['type']=='Operation']['cost_cat'].tolist()]
        #Turn each cost category into a column
        df = df.pivot_table(index=['year'], columns='cost_cat', values='Cost (Bil $)')
        #Add rows for all years (including 20 years after end year) and fill
        full_yrs = list(range(df.index.min() - 19, df.index.max() + 21))
        df = df.reindex(full_yrs)
        #For capital costs, multiply by CRF to annualize, and sum over previous 20 years.
        #This requires 20 years before 2010 to sum properly, and we need to shift capital dataframe down
        #so that capital payments start in the year after the investment was made
        CRF = d*(1+d)**20/((1+d)**20 - 1)
        df[cap_type_ls] = df[cap_type_ls].shift().fillna(0)*CRF
        df[cap_type_ls] = df[cap_type_ls].rolling(20).sum()
        #Remove years before 2010
        full_yrs = list(range(df.index.min() + 19, df.index.max() + 1))
        df = df.reindex(full_yrs)
        #For operation costs, simply fill missing years with model year values.
        df[op_type_ls] = df[op_type_ls].fillna(method='ffill')
        #The final year should only include capital payments because operation payments last for 20 yrs starting
        #in the model year, whereas capital payments last for 20 yrs starting in the year after the model year.
        df.loc[df.index.max(), op_type_ls] = 0
        df = df.fillna(0)
        df = pd.melt(df.reset_index(), id_vars=['year'], value_vars=cap_type_ls + op_type_ls, var_name='cost_cat', value_name= 'Cost (Bil $)')

    #Add Dicounted Cost column
    df['Discounted Cost (Bil $)'] = df['Cost (Bil $)'] / (1 + d)**(df['year'] - y0)
    return df

def pre_abatement_cost(dfs, **kW):
    d = float(core.GL['widgets']['var_discount_rate'].value)
    y0 = int(core.GL['widgets']['var_pv_year'].value)

    df_sc = pre_systemcost(dfs)
    df_sc = sum_over_cols(df_sc, group_cols=['year'], drop_cols=['cost_cat','Discounted Cost (Bil $)'])
    df_sc['type'] = 'Cost (Bil $)'
    df_sc.rename(columns={'Cost (Bil $)':'val'}, inplace=True)

    df_co2 = dfs['emit']
    df_co2.rename(columns={'CO2 (MMton)':'val'}, inplace=True)
    df_co2['val'] = df_co2['val'] * 1e-9 #converting from million to billion metric tons
    df_co2['type'] = 'CO2 (Bil metric ton)'

    df = pd.concat([df_sc, df_co2],sort=False,ignore_index=True)
    df['disc val'] = df['val'] / (1 + d)**(df['year'] - y0)
    return df


def map_i_to_n(df, **kw):
    df_hier = pd.read_csv(this_dir_path + '/in/reeds2/region_map.csv')
    dict_hier = dict(zip(df_hier['s'], df_hier['n']))
    df.loc[df['region'].isin(dict_hier.keys()), 'region'] = df['region'].map(dict_hier)
    df.rename(columns={'region':'n'}, inplace=True)
    if 'groupsum' in kw:
        df = df.groupby(kw['groupsum'], sort=False, as_index=False).sum()
    return df

def remove_n(df, **kw):
    df = df[~df['region'].astype(str).str.startswith('p')].copy()
    df['region'] = df['region'].map(lambda x: x.lstrip('s'))
    df.rename(columns={'region':'i'}, inplace=True)
    return df

def pre_val_streams(dfs, **kw):
    if 'investment_only' in kw:
        #Analyze only investment years
        #The first attempt of this was to use the ratio of new vs cumulative capacity in a vintage, but this has issues
        #because of the mismatch in regionality between capacity and generation, meaning ba-level capacity ratio may not
        #even out value streams.
        #First, use the capacity/investment linking equations with the investment and capacity variables to find the
        #scaling factors between investment and capacity value streams
        cols = ['tech', 'vintage', 'n', 'year']
        inv_vars = ['inv','invrefurb']
        cum_vars = ['gen','cap','opres','storage_in']
        linking_eqs = ['eq_cap_new_noret','eq_cap_new_retub','eq_cap_new_retmo'] #eq_cap_new_retmo also includes previous year's CAP, is this bad?!
        df_vs_links = dfs['vs'][dfs['vs']['con_name'].isin(linking_eqs)].copy()
        df_vs_inv = df_vs_links[df_vs_links['var_name'].isin(inv_vars)].copy()
        df_vs_cap = df_vs_links[df_vs_links['var_name'] == 'cap'].copy()
        df_vs_inv = sum_over_cols(df_vs_inv, group_cols=cols, drop_cols=['var_name','con_name'])
        df_vs_cap = sum_over_cols(df_vs_cap, group_cols=cols, drop_cols=['var_name','con_name'])
        #merge left with df_vs_inv so that we're only looking at cumulative value streams in investment years.
        df_scale = pd.merge(left=df_vs_inv, right=df_vs_cap, how='left', on=cols, sort=False)
        df_scale['mult'] = df_scale['$_x'] / df_scale['$_y'] * -1
        df_scale = df_scale[cols + ['mult']]
        #Gather cumulative value streams. NEED TO ADD TRANSMISSION
        df_cum = dfs['vs'][dfs['vs']['var_name'].isin(cum_vars)].copy()
        #Left merge with df_scale to keep only the cumulative streams in investment years
        df_cum = pd.merge(left=df_scale, right=df_cum, how='left', on=cols, sort=False)
        #Scale the cumulative value streams
        df_cum['$'] = df_cum['$'] * df_cum['mult']
        df_cum.drop(['mult'], axis='columns',inplace=True)
        #Concatenate modified cumulative value streams with the rest of the value streams
        dfs['vs'] = dfs['vs'][~dfs['vs']['var_name'].isin(cum_vars)].copy()
        dfs['vs'] = pd.concat([dfs['vs'], df_cum],sort=False,ignore_index=True)
        #Adjust generation based on the same scaling factor. Not sure this is exactly right, but if
        #value streams for GEN have scaled, it makes sense to attribute this to quantity of energy being scaled,
        #rather than prices changing.
        dfs['gen'] = pd.merge(left=df_scale, right=dfs['gen'], how='left', on=cols, sort=False)
        dfs['gen']['MWh'] = dfs['gen']['MWh'] * dfs['gen']['mult']
        dfs['gen'].drop(['mult'], axis='columns',inplace=True)

    #apply inflation
    dfs['vs']['$'] = inflate_series(dfs['vs']['$'])
    #Use pvf_capital to convert to present value as of data year (model year for CAP and GEN but investment year for INV,
    #although i suppose certain equations, e.g. eq_cap_new_retmo also include previous year's CAP).
    df = pd.merge(left=dfs['vs'], right=dfs['pvf_cap'], how='left', on=['year'], sort=False)
    df['Bulk $'] = df['$'] / dfs['cost_scale'].iloc[0,0] / df['pvfcap']
    df.drop(['pvfcap', '$'], axis='columns',inplace=True)
    #Preprocess gen: convert from annual MWh to bulk MWh present value as of data year
    df_gen = dfs['gen'].groupby(['tech','vintage','n','year'], sort=False, as_index =False).sum()
    df_gen = pd.merge(left=df_gen, right=dfs['pvf_cap'], how='left', on=['year'], sort=False)
    df_gen = pd.merge(left=df_gen, right=dfs['pvf_onm'], how='left', on=['year'], sort=False)
    df_gen['MWh'] = df_gen['MWh'] * df_gen['pvfonm'] / df_gen['pvfcap'] #This converts to bulk MWh present value as of data year
    df_gen.rename(columns={'MWh': 'Bulk $'}, inplace=True) #So we can concatenate
    df_gen['var_name'] = 'MWh'
    df_gen['con_name'] = 'MWh'
    df_gen.drop(['pvfcap', 'pvfonm'], axis='columns',inplace=True)
    #Preprocess new capacity: map i to n, convert from MW to kW, reformat columns to concatenate
    df_cap = map_i_to_n(dfs['cap'])
    df_cap =  df_cap.groupby(['tech','vintage','n','year'], sort=False, as_index =False).sum()
    df_cap['MW'] = df_cap['MW'] * 1000 #Converting to kW
    df_cap.rename(columns={'MW': 'Bulk $'}, inplace=True) #So we can concatenate
    df_cap['var_name'] = 'kW'
    df_cap['con_name'] = 'kW'
    df = pd.concat([df, df_gen, df_cap],sort=False,ignore_index=True)
    #Add discounted $ using interface year
    d = float(core.GL['widgets']['var_discount_rate'].value)
    y0 = int(core.GL['widgets']['var_pv_year'].value)
    df['Bulk $ Dis'] = df['Bulk $'] / (1 + d)**(df['year'] - y0) #This discounts $, MWh, and kW, important for NVOE, NVOC, LCOE, etc.

    df['tech, vintage'] = df['tech'] + ', ' + df['vintage']
    df['var, con'] = df['var_name'] + ', ' + df['con_name']
    #Make adjusted con_name column where all _obj are replaced with var_name, _obj
    df['con_adj'] = df['con_name']
    df.loc[df['con_name'] == '_obj', 'con_adj'] = df.loc[df['con_name'] == '_obj', 'var, con']
    return df


def pre_val_streams_old(df, **kw):
    df_not_dol = df[df['con_name'].isin(['mwh', 'kw'])].copy()
    df_dol = df[~df['con_name'].isin(['mwh', 'kw'])].copy()
    #apply inflation and annualize
    df_dol['value'] = inflate_series(df_dol['value']) * CRF_reeds
    #adjust capacity of PV???
    df = pd.concat([df_not_dol, df_dol],sort=False,ignore_index=True)
    return df

def pre_reduced_cost(df, **kw):
    df['irbv'] = df['tech'] + ' | ' + df['region'] + ' | ' + df['bin'] + ' | ' + df['variable']
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

def pre_curt_new(dfs, **kw):
    df = pd.merge(left=dfs['gen_uncurt'], right=dfs['curt_rate'], how='left',on=['tech', 'rr', 'timeslice', 'year'], sort=False)
    df['Curt Rate']=df['Curt Rate'].fillna(0)
    return df

def pre_cc_new(dfs, **kw):
    df = pd.merge(left=dfs['cap'], right=dfs['cc'], how='left',on=['tech', 'rr', 'season', 'year'], sort=False)
    df['CC Rate']=df['CC Rate'].fillna(0)
    return df

def pre_firm_cap(dfs, **kw):
    #aggregate capacity to ba-level
    df_cap = map_i_to_n(dfs['cap'], groupsum=['tech','n','year'])
    #Add seasons to capacity dataframe
    dftemp = pd.DataFrame({'season':dfs['firmcap']['season'].unique().tolist()})
    dftemp['temp'] = 1
    df_cap['temp'] = 1
    df_cap = pd.merge(left=df_cap, right=dftemp, how='left',on=['temp'], sort=False)
    df_cap.drop(columns=['temp'],inplace=True)
    df = pd.merge(left=df_cap, right=dfs['firmcap'], how='left',on=['tech', 'n', 'year','season'], sort=False)
    df = df.fillna(0)
    df[['Capacity (GW)','Firm Capacity (GW)']] = df[['Capacity (GW)','Firm Capacity (GW)']] * 1e-3
    df['Capacity Credit'] = df['Firm Capacity (GW)'] / df['Capacity (GW)']
    return df

def pre_curt(dfs, **kw):
    df = pd.merge(left=dfs['gen_uncurt'], right=dfs['gen'], how='left',on=['tech', 'vintage', 'n', 'year'], sort=False)
    df['MWh']=df['MWh'].fillna(0)
    df['Curt Rate'] = 1 - df['MWh']/df['MWh uncurt']
    df_re_n = sum_over_cols(dfs['gen_uncurt'], group_cols=['n','year'], drop_cols=['tech','vintage'])
    df_re_nat = sum_over_cols(dfs['gen_uncurt'], group_cols=['year'], drop_cols=['tech','vintage','n'])
    df_load_nat = sum_over_cols(dfs['load'], group_cols=['year'], drop_cols=['n'])
    df_vrepen_n = pd.merge(left=dfs['load'], right=df_re_n, how='left',on=['n', 'year'], sort=False)
    df_vrepen_n['VRE penetration n'] = df_vrepen_n['MWh uncurt'] / df_vrepen_n['MWh load']
    df_vrepen_n = df_vrepen_n[['n','year','VRE penetration n']]
    df_vrepen_nat = pd.merge(left=df_load_nat, right=df_re_nat, how='left',on=['year'], sort=False)
    df_vrepen_nat['VRE penetration nat'] = df_vrepen_nat['MWh uncurt'] / df_vrepen_nat['MWh load']
    df_vrepen_nat = df_vrepen_nat[['year','VRE penetration nat']]
    df = pd.merge(left=df, right=df_vrepen_n, how='left',on=['n', 'year'], sort=False)
    df = pd.merge(left=df, right=df_vrepen_nat, how='left',on=['year'], sort=False)
    return df

def pre_curt_iter(dfs, **kw):
    df_gen = dfs['gen_uncurt']
    df_curt = dfs['curt']
    df_gen = df_gen[df_gen['tech'].isin(df_curt['tech'].unique())].copy()
    df_gen['type'] = 'gen'
    df_curt['type'] = 'curt'
    df = pd.concat([df_gen, df_curt],sort=False,ignore_index=True)
    return df

def pre_cc_iter(dfs, **kw):
    df_cap = dfs['cap']
    df_cap_firm = dfs['cap_firm']
    df_cap = df_cap[df_cap['tech'].isin(df_cap_firm['tech'].unique())].copy()
    df_cap['type'] = 'cap'
    df_cap_firm['type'] = 'cap_firm'
    seasons = list(df_cap_firm['season'].unique())
    df_season = pd.DataFrame({'season':seasons,'temp':[1]*len(seasons)})
    df_cap['temp'] = 1
    df_cap = pd.merge(left=df_cap, right=df_season, how='left',on=['temp'], sort=False)
    df_cap.drop(['temp'], axis='columns',inplace=True)
    df = pd.concat([df_cap, df_cap_firm],sort=False,ignore_index=True)
    return df

def pre_cf(dfs, **kw):
    df = pd.merge(left=dfs['cap'], right=dfs['gen'], how='left',on=['tech', 'vintage', 'n', 'year'], sort=False)
    df['MWh']=df['MWh'].fillna(0)
    df['CF'] = df['MWh']/(df['MW']*8760)
    return df

def pre_prices(dfs, **kw):
    #Apply inflation
    dfs['p']['p'] = inflate_series(dfs['p']['p'])
    #Join prices and quantities
    df = pd.merge(left=dfs['q'], right=dfs['p'], how='left', on=['type', 'subtype', 'n', 'timeslice', 'year'], sort=False)
    df['p'].fillna(0, inplace=True)
    #Calculate $
    df['$'] = df['p'] * df['q']
    df.drop(['p', 'q'], axis='columns',inplace=True)
    #Concatenate quantities
    df_q = dfs['q']
    df_q.rename(columns={'q':'$'}, inplace=True)
    df_q['type'] = 'q_' + df_q['type']
    df = pd.concat([df, df_q],sort=False,ignore_index=True)
    return df

def pre_ng_price(dfs, **kw):
    #Apply inflation
    dfs['p']['p'] = inflate_series(dfs['p']['p'])
    #Join prices and quantities
    df = pd.merge(left=dfs['q'], right=dfs['p'], how='left', on=['census', 'year'], sort=False)
    df['p'].fillna(0, inplace=True)
    return df

def add_joint_locations_col(df, **kw):
    df[kw['new']] = df[kw['col1']] + '-' + df[kw['col2']]
    return df
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------

#2. Columns metadata. These are columns that are referenced in the Results section below.
#This is where joins, maps, and styles are applied for the columns.
#For 'style', colors are in hex, but descriptions are given (see http://www.color-hex.com/color-names.html).
columns_meta = {
    'tech':{
        'type':'string',
        'map': this_dir_path + '/in/reeds2/tech_map.csv',
        'style': this_dir_path + '/in/reeds2/tech_style.csv',
    },
    'class':{
        'type':'string',
    },
    'region':{
        'type':'string',
    },
    'rr':{
        'type':'string',
        'join': this_dir_path + '/in/reeds2/hierarchy_rr.csv',
    },
    'i':{
        'type':'string',
        'join': this_dir_path + '/in/hierarchy.csv',
    },
    'n':{
        'type':'string',
        'join': this_dir_path + '/in/hierarchy.csv',
    },
    'timeslice':{
        'type':'string',
        'map': this_dir_path + '/in/reeds2/m_map.csv',
        'style': this_dir_path + '/in/reeds2/m_style.csv',
    },
    'year':{
        'type':'number',
        'filterable': True,
        'seriesable': True,
        'y-allow': False,
    },
    'iter':{
        'type':'string',
    },
    'icrb':{
        'type': 'string',
        'filterable': False,
        'seriesable': False,
    },
    'irbv':{
        'type': 'string',
        'filterable': False,
        'seriesable': False,
    },
    'cost_cat':{
        'type':'string',
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
        {'file':'cap.csv',
        'columns': ['tech', 'region', 'year', 'Capacity (GW)'],
        'preprocess': [
            {'func': map_i_to_n, 'args': {}},
            {'func': scale_column, 'args': {'scale_factor': .001, 'column':'Capacity (GW)'}},
        ],
        'index': ['tech', 'n', 'year'],
        'presets': collections.OrderedDict((
            ('Stacked Area',{'x':'year', 'y':'Capacity (GW)', 'series':'tech', 'explode':'scenario', 'chart_type':'Area'}),
            ('Stacked Bars',{'x':'year', 'y':'Capacity (GW)', 'series':'tech', 'explode':'scenario', 'chart_type':'Bar', 'bar_width':'1.75'}),
            ('Explode By Tech',{'x':'year', 'y':'Capacity (GW)', 'series':'scenario', 'explode':'tech', 'chart_type':'Line'}),
            ('PCA Map Final by Tech',{'x':'n', 'y':'Capacity (GW)', 'explode':'scenario', 'explode_group':'tech', 'chart_type':'Map', 'filter': {'year':'last'}}),
            ('State Map Final by Tech',{'x':'st', 'y':'Capacity (GW)', 'explode':'scenario', 'explode_group':'tech', 'chart_type':'Map', 'filter': {'year':'last'}}),
        )),
        }
    ),

    ('New Annual Capacity BA (GW)',
        {'file':'cap_new_ann.csv',
        'columns': ['tech', 'region', 'year', 'Capacity (GW)'],
        'preprocess': [
            {'func': map_i_to_n, 'args': {}},
            {'func': scale_column, 'args': {'scale_factor': .001, 'column':'Capacity (GW)'}},
        ],
        'index': ['tech', 'n', 'year'],
        'presets': collections.OrderedDict((
            ('Stacked Area',{'x':'year', 'y':'Capacity (GW)', 'series':'tech', 'explode':'scenario', 'chart_type':'Area'}),
            ('Stacked Bars',{'x':'year', 'y':'Capacity (GW)', 'series':'tech', 'explode':'scenario', 'chart_type':'Bar', 'bar_width':'1.75'}),
            ('Explode By Tech',{'x':'year', 'y':'Capacity (GW)', 'series':'scenario', 'explode':'tech', 'chart_type':'Line'}),
            ('PCA Map Final by Tech',{'x':'n', 'y':'Capacity (GW)', 'explode':'scenario', 'explode_group':'tech', 'chart_type':'Map', 'filter': {'year':'last'}}),
            ('State Map Final by Tech',{'x':'st', 'y':'Capacity (GW)', 'explode':'scenario', 'explode_group':'tech', 'chart_type':'Map', 'filter': {'year':'last'}}),
        )),
        }
    ),

    ('Annual Retirements BA (GW)',
        {'file':'ret_ann.csv',
        'columns': ['tech', 'region', 'year', 'Capacity (GW)'],
        'preprocess': [
            {'func': map_i_to_n, 'args': {}},
            {'func': scale_column, 'args': {'scale_factor': .001, 'column':'Capacity (GW)'}},
        ],
        'index': ['tech', 'n', 'year'],
        'presets': collections.OrderedDict((
            ('Stacked Area',{'x':'year', 'y':'Capacity (GW)', 'series':'tech', 'explode':'scenario', 'chart_type':'Area'}),
            ('Stacked Bars',{'x':'year', 'y':'Capacity (GW)', 'series':'tech', 'explode':'scenario', 'chart_type':'Bar', 'bar_width':'1.75'}),
            ('Explode By Tech',{'x':'year', 'y':'Capacity (GW)', 'series':'scenario', 'explode':'tech', 'chart_type':'Line'}),
            ('PCA Map Final by Tech',{'x':'n', 'y':'Capacity (GW)', 'explode':'scenario', 'explode_group':'tech', 'chart_type':'Map', 'filter': {'year':'last'}}),
            ('State Map Final by Tech',{'x':'st', 'y':'Capacity (GW)', 'explode':'scenario', 'explode_group':'tech', 'chart_type':'Map', 'filter': {'year':'last'}}),
        )),
        }
    ),

    ('Capacity Resource Region (GW)',
        {'file':'cap.csv',
        'columns': ['tech', 'region', 'year', 'Capacity (GW)'],
        'preprocess': [
            {'func': remove_n, 'args': {}},
            {'func': scale_column, 'args': {'scale_factor': .001, 'column':'Capacity (GW)'}},
        ],
        'index': ['tech', 'i', 'year'],
        'presets': collections.OrderedDict((
            ('Stacked Area',{'x':'year', 'y':'Capacity (GW)', 'series':'tech', 'explode':'scenario', 'chart_type':'Area'}),
            ('Stacked Bars',{'x':'year', 'y':'Capacity (GW)', 'series':'tech', 'explode':'scenario', 'chart_type':'Bar', 'bar_width':'1.75'}),
            ('Explode By Tech',{'x':'year', 'y':'Capacity (GW)', 'series':'scenario', 'explode':'tech', 'chart_type':'Line'}),
            ('RR Map Final by Tech',{'x':'i', 'y':'Capacity (GW)', 'explode':'scenario', 'explode_group':'tech', 'chart_type':'Map', 'filter': {'year':'last'}}),
            ('RR Map Final Wind',{'x':'i', 'y':'Capacity (GW)', 'explode':'scenario', 'chart_type':'Map', 'filter': {'year':'last', 'tech':['wind-ons', 'wind-ofs']}}),
        )),
        }
    ),


    ('Generation BA (TWh)',
        {'file':'gen_ann.csv',
        'columns': ['tech', 'region', 'year', 'Generation (TWh)'],
        'preprocess': [
            {'func': map_i_to_n, 'args': {}},
            {'func': scale_column, 'args': {'scale_factor': 1e-6, 'column':'Generation (TWh)'}},
        ],
        'index': ['tech', 'n', 'year'],
        'presets': collections.OrderedDict((
            ('Stacked Area',{'x':'year', 'y':'Generation (TWh)', 'series':'tech', 'explode':'scenario', 'chart_type':'Area'}),
            ('Stacked Bars',{'x':'year', 'y':'Generation (TWh)', 'series':'tech', 'explode':'scenario', 'chart_type':'Bar', 'bar_width':'1.75'}),
            ('Explode By Tech',{'x':'year', 'y':'Generation (TWh)', 'series':'scenario', 'explode':'tech', 'chart_type':'Line'}),
            ('PCA Map Final by Tech',{'x':'n', 'y':'Generation (TWh)', 'explode':'scenario', 'explode_group':'tech', 'chart_type':'Map', 'filter': {'year':'last'}}),
            ('State Map Final by Tech',{'x':'st', 'y':'Generation (TWh)', 'explode':'scenario', 'explode_group':'tech', 'chart_type':'Map', 'filter': {'year':'last'}}),
        )),
        }
    ),

    ('Gen by timeslice national (GW)',
        {'file':'gen_h.csv',
        'columns': ['tech', 'region', 'timeslice', 'year', 'Generation (GW)'],
        'index': ['tech', 'year', 'timeslice'],
        'preprocess': [
            {'func': map_i_to_n, 'args': {}},
            {'func': sum_over_cols, 'args': {'drop_cols': ['n'], 'group_cols': ['tech', 'year', 'timeslice']}},
            {'func': scale_column, 'args': {'scale_factor': .001, 'column':'Generation (GW)'}},
        ],
        'presets': collections.OrderedDict((
            ('Stacked Bars Final',{'x':'timeslice', 'y':'Generation (GW)', 'series':'tech', 'explode':'scenario', 'chart_type':'Bar', 'filter': {'year':'last'}}),
        )),
        }
    ),

    ('Operating Reserves (TW-h)',
        {'file':'opres_supply.csv',
        'columns': ['ortype', 'tech', 'region', 'year', 'Reserves (TW-h)'],
        'index': ['ortype', 'tech', 'year'],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': 1e-6, 'column':'Reserves (TW-h)'}},
        ],
        'presets': collections.OrderedDict((
            ('Stacked Bars',{'x':'year', 'y':'Reserves (TW-h)', 'series':'tech', 'explode':'scenario', 'explode_group':'ortype', 'chart_type':'Bar', 'bar_width':'1.75', }),
        )),
        }
    ),

    ('Operating Reserves by Timeslice National (GW)',
        {'file':'opres_supply_h.csv',
        'columns': ['ortype', 'tech', 'region', 'timeslice', 'year', 'Reserves (GW)'],
        'index': ['ortype', 'tech', 'year', 'timeslice'],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': .001, 'column':'Reserves (GW)'}},
        ],
        'presets': collections.OrderedDict((
            ('Stacked Bars Final',{'x':'timeslice', 'y':'Reserves (GW)', 'series':'tech', 'explode':'scenario', 'explode_group':'ortype', 'chart_type':'Bar', 'filter': {'year':'last'}}),
        )),
        }
    ),

    ('Firm Capacity (GW)',
        {'sources': [
            {'name': 'firmcap', 'file': 'cap_firm.csv', 'columns': ['tech', 'n', 'season', 'year', 'Firm Capacity (GW)']},
            {'name': 'cap', 'file': 'cap.csv', 'columns': ['tech', 'region', 'year', 'Capacity (GW)']},
        ],
        'index': ['tech', 'n', 'season', 'year'],
        'preprocess': [
            {'func': pre_firm_cap, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Stacked Bars',{'x':'year', 'y':'Firm Capacity (GW)', 'series':'tech', 'explode':'scenario', 'explode_group':'season', 'chart_type':'Bar', 'bar_width':'1.75'}),
            ('Average Capacity Credit',{'x':'year', 'y':'Capacity Credit', 'y_agg':'Weighted Ave', 'y_weight':'Capacity (GW)', 'series':'scenario', 'explode':'season', 'explode_group':'tech', 'chart_type':'Line'}),
        )),
        }
    ),

    ('CO2 Emissions National (MMton)',
        {'file':'emit_nat.csv',
        'columns': ['year', 'CO2 (MMton)'],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': 1e-6, 'column':'CO2 (MMton)'}},
        ],
        'index': ['year'],
        'presets': collections.OrderedDict((
            ('Scenario Lines Over Time',{'x':'year', 'y':'CO2 (MMton)', 'series':'scenario', 'chart_type':'Line'}),
        )),
        }
    ),

    ('CO2 Emissions BA (MMton)',
        {'file':'emit_r.csv',
        'columns': ['n', 'year', 'CO2 (MMton)'],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': 1e-6, 'column':'CO2 (MMton)'}},
        ],
        'index': ['n', 'year'],
        'presets': collections.OrderedDict((
            ('Final BA Map',{'x':'n', 'y':'CO2 (MMton)', 'explode':'scenario', 'chart_type':'Map', 'filter': {'year':'last'}}),
        )),
        }
    ),

    ('Natural Gas Price ($/MMBtu)',
        {'sources': [
            {'name': 'p', 'file': 'repgasprice.csv', 'columns': ['census', 'year', 'p']},
            {'name': 'q', 'file': 'repgasquant.csv', 'columns': ['census', 'year', 'q']},
        ],
        'preprocess': [
            {'func': pre_ng_price, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Boxplot',{'chart_type':'Dot', 'x':'year', 'y':'p', 'y_agg':'None', 'range':'Boxplot', 'explode':'scenario', 'sync_axes':'No', 'circle_size':r'3', 'bar_width':r'1.75', }),
            ('Weighted Ave',{'chart_type':'Line', 'x':'year', 'y':'p', 'y_agg':'Weighted Ave', 'y_weight':'q', 'series':'scenario', 'sync_axes':'No', }),
        )),
        }
    ),

    ('Requirement Prices and Quantities',
        {'sources': [
            {'name': 'p', 'file': 'reqt_price.csv', 'columns': ['type', 'subtype', 'n', 'timeslice', 'year', 'p']},
            {'name': 'q', 'file': 'reqt_quant.csv', 'columns': ['type', 'subtype', 'n', 'timeslice', 'year', 'q']},
        ],
        'preprocess': [
            {'func': pre_prices, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Energy Price Lines ($/MWh)',{'x':'year', 'y':'$', 'series':'scenario', 'explode': 'type', 'chart_type':'Line', 'adv_op':'Ratio', 'adv_col':'type', 'adv_col_base':'q_load', 'filter': {'type':['load','q_load']}}),
            ('OpRes Price Lines ($/MW-h)',{'x':'year', 'y':'$', 'series':'scenario', 'explode': 'subtype', 'explode_group':'type', 'chart_type':'Line', 'adv_op':'Ratio', 'adv_col':'type', 'adv_col_base':'q_oper_res', 'filter': {'type':['oper_res','q_oper_res']}}),
            ('ResMarg Price Lines ($/kW-yr)',{'x':'year', 'y':'$', 'series':'scenario', 'explode': 'type', 'chart_type':'Line', 'adv_op':'Ratio', 'adv_col':'type', 'adv_col_base':'q_res_marg_ann', 'filter': {'type':['res_marg_ann','q_res_marg_ann']}}),
            ('ResMarg Season Price Lines ($/kW-yr)',{'x':'year', 'y':'$', 'series':'scenario', 'explode': 'timeslice', 'explode_group':'type', 'chart_type':'Line', 'adv_op':'Ratio', 'adv_col':'type', 'adv_col_base':'q_res_marg', 'filter': {'type':['res_marg','q_res_marg']}}),
            ('Energy Price by Timeslice Final ($/MWh)',{'x':'timeslice', 'y':'$', 'series':'type', 'explode':'scenario', 'chart_type':'Bar', 'adv_op':'Ratio', 'adv_col':'type', 'adv_col_base':'q_load', 'filter': {'type':['load','q_load'], 'year':'last'}}),
            ('OpRes Price by Timeslice Final ($/MW-h)',{'x':'timeslice', 'y':'$', 'series':'type', 'explode':'subtype', 'explode_group':'scenario', 'chart_type':'Bar', 'adv_op':'Ratio', 'adv_col':'type', 'adv_col_base':'q_oper_res', 'filter': {'type':['oper_res','q_oper_res'], 'year':'last'}}),
            ('Energy Price Final BA Map ($/MWh)',{'x':'n', 'y':'$', 'explode': 'scenario', 'explode_group': 'type', 'chart_type':'Map', 'adv_op':'Ratio', 'adv_col':'type', 'adv_col_base':'q_load', 'filter': {'type':['load','q_load'], 'year':'last'}}),
            ('All-in Price ($/MWh)',{'x':'year', 'y':'$', 'series':'type', 'explode': 'scenario', 'chart_type':'Bar', 'bar_width':'1.75', 'adv_op':'Ratio', 'adv_col':'type', 'adv_col_base':'q_load', 'filter': {'type':['load','res_marg','oper_res','state_rps','q_load']}}),
        )),
        }
    ),

    ('Requirement Prices and Quantities OLD',
        {'sources': [
            {'name': 'p', 'file': 'reqt_price.csv', 'columns': ['type', 'subtype', 'n', 'timeslice', 'year', 'p']},
            {'name': 'q', 'file': 'reqt_quant.csv', 'columns': ['type', 'subtype', 'n', 'timeslice', 'year', 'q']},
        ],
        'preprocess': [
            {'func': pre_prices, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Energy Price Lines ($/MWh)',{'x':'year', 'y':'$', 'series':'scenario', 'explode': 'type', 'chart_type':'Line', 'adv_op':'Ratio', 'adv_col':'type', 'adv_col_base':'q_load', 'filter': {'type':['load','q_load']}}),
            ('OpRes Price Lines ($/MW-h)',{'x':'year', 'y':'$', 'series':'scenario', 'explode': 'subtype', 'explode_group':'type', 'chart_type':'Line', 'adv_op':'Ratio', 'adv_col':'type', 'adv_col_base':'q_oper_res', 'filter': {'type':['oper_res','q_oper_res']}}),
            ('ResMarg Price Lines ($/kW-yr)',{'x':'year', 'y':'$', 'series':'scenario', 'explode': 'type', 'chart_type':'Line', 'adv_op':'Ratio', 'adv_col':'type', 'adv_col_base':'q_res_marg', 'filter': {'type':['res_marg','q_res_marg'], 'timeslice':['Annual']}}),
            ('ResMarg Season Price Lines ($/kW-yr)',{'x':'year', 'y':'$', 'series':'scenario', 'explode': 'timeslice', 'explode_group':'type', 'chart_type':'Line', 'adv_op':'Ratio', 'adv_col':'type', 'adv_col_base':'q_res_marg', 'filter': {'type':['res_marg','q_res_marg'], 'timeslice':['Summer','Fall','Winter','Spring']}}),
            ('Energy Price by Timeslice Final ($/MWh)',{'x':'timeslice', 'y':'$', 'series':'type', 'explode':'scenario', 'chart_type':'Bar', 'adv_op':'Ratio', 'adv_col':'type', 'adv_col_base':'q_load', 'filter': {'type':['load','q_load'], 'year':'last'}}),
            ('OpRes Price by Timeslice Final ($/MW-h)',{'x':'timeslice', 'y':'$', 'series':'type', 'explode':'subtype', 'explode_group':'scenario', 'chart_type':'Bar', 'adv_op':'Ratio', 'adv_col':'type', 'adv_col_base':'q_oper_res', 'filter': {'type':['oper_res','q_oper_res'], 'year':'last'}}),
            ('Energy Price Final BA Map ($/MWh)',{'x':'n', 'y':'$', 'explode': 'scenario', 'explode_group': 'type', 'chart_type':'Map', 'adv_op':'Ratio', 'adv_col':'type', 'adv_col_base':'q_load', 'filter': {'type':['load','q_load'], 'year':'last'}}),
            ('All-in Price ($/MWh)',{'x':'year', 'y':'$', 'series':'type', 'explode': 'scenario', 'chart_type':'Bar', 'bar_width':'1.75', 'adv_op':'Ratio', 'adv_col':'type', 'adv_col_base':'q_load', 'filter': {'type':['load','res_marg','oper_res','q_load'], 'timeslice':{'exclude': ['Annual']}}}),
        )),
        }
    ),

    ('Sys Cost Annualized (Bil $)',
        {'sources': [
            {'name': 'sc', 'file': 'systemcost.csv', 'columns': ['cost_cat', 'year', 'Cost (Bil $)']},
        ],
        'index': ['cost_cat', 'year'],
        'preprocess': [
            {'func': pre_systemcost, 'args': {'annualize':True}},
        ],
        'presets': collections.OrderedDict((
            ('Total Discounted',{'x':'scenario','y':'Discounted Cost (Bil $)','series':'cost_cat','chart_type':'Bar', 'filter': {'cost_cat':{'exclude':costs_orig_inv}}}),
            ('Total Discounted No Pol',{'x':'scenario','y':'Discounted Cost (Bil $)','series':'cost_cat','chart_type':'Bar', 'filter': {'cost_cat':{'exclude':costs_pol_inv}}}),
            ('2018-end Discounted',{'x':'scenario','y':'Discounted Cost (Bil $)','series':'cost_cat','chart_type':'Bar', 'filter': {'cost_cat':{'exclude':costs_orig_inv}, 'year': {'start':2018}}}),
            ('2018-end Discounted No Pol',{'x':'scenario','y':'Discounted Cost (Bil $)','series':'cost_cat','chart_type':'Bar', 'filter': {'cost_cat':{'exclude':costs_pol_inv}, 'year': {'start':2018}}}),
            ('Discounted by Year',{'x':'year','y':'Discounted Cost (Bil $)','series':'cost_cat','explode':'scenario','chart_type':'Bar', 'bar_width':'1.75', 'filter': {'cost_cat':{'exclude':costs_orig_inv}}}),
            ('Discounted by Year No Pol',{'x':'year','y':'Discounted Cost (Bil $)','series':'cost_cat','explode':'scenario','chart_type':'Bar', 'bar_width':'1.75', 'filter': {'cost_cat':{'exclude':costs_pol_inv}}}),
            ('Undiscounted by Year',{'x':'year','y':'Cost (Bil $)','series':'cost_cat','explode':'scenario','chart_type':'Bar', 'bar_width':'1.75', 'filter': {'cost_cat':{'exclude':costs_orig_inv}}}),
            ('Undiscounted by Year No Pol',{'x':'year','y':'Cost (Bil $)','series':'cost_cat','explode':'scenario','chart_type':'Bar', 'bar_width':'1.75', 'filter': {'cost_cat':{'exclude':costs_pol_inv}}}),
        )),
        }
    ),

    ('CO2 Abatement Cost ($/metric ton)',
        {'sources': [
            {'name': 'sc', 'file': 'systemcost.csv', 'columns': ['cost_cat', 'year', 'Cost (Bil $)']},
            {'name': 'emit', 'file': 'emit_nat.csv', 'columns': ['year', 'CO2 (MMton)']},
        ],
        'index': ['year','type'],
        'preprocess': [
            {'func': pre_abatement_cost, 'args': {'annualize':True}},
        ],
        'presets': collections.OrderedDict((
        )),
        }
    ),

    ('Sys Cost Bulk (Bil $)',
        {'sources': [
            {'name': 'sc', 'file': 'systemcost_bulk.csv', 'columns': ['cost_cat', 'year', 'Cost (Bil $)']},
        ],
        'index': ['cost_cat', 'year'],
        'preprocess': [
            {'func': pre_systemcost, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Total Discounted',{'x':'scenario','y':'Discounted Cost (Bil $)','series':'cost_cat','chart_type':'Bar', 'filter': {'cost_cat':{'exclude':costs_orig_inv}}}),
            ('Total Discounted No Pol',{'x':'scenario','y':'Discounted Cost (Bil $)','series':'cost_cat','chart_type':'Bar', 'filter': {'cost_cat':{'exclude':costs_pol_inv}}}),
            ('2018-end Discounted',{'x':'scenario','y':'Discounted Cost (Bil $)','series':'cost_cat','chart_type':'Bar', 'filter': {'cost_cat':{'exclude':costs_orig_inv}, 'year': {'start':2018}}}),
            ('2018-end Discounted No Pol',{'x':'scenario','y':'Discounted Cost (Bil $)','series':'cost_cat','chart_type':'Bar', 'filter': {'cost_cat':{'exclude':costs_pol_inv}, 'year': {'start':2018}}}),
            ('Discounted by Year',{'x':'year','y':'Discounted Cost (Bil $)','series':'cost_cat','explode':'scenario','chart_type':'Bar', 'bar_width':'1.75', 'filter': {'cost_cat':{'exclude':costs_orig_inv}}}),
            ('Discounted by Year No Pol',{'x':'year','y':'Discounted Cost (Bil $)','series':'cost_cat','explode':'scenario','chart_type':'Bar', 'bar_width':'1.75', 'filter': {'cost_cat':{'exclude':costs_pol_inv}}}),
            ('Undiscounted by Year',{'x':'year','y':'Cost (Bil $)','series':'cost_cat','explode':'scenario','chart_type':'Bar', 'bar_width':'1.75', 'filter': {'cost_cat':{'exclude':costs_orig_inv}}}),
            ('Undiscounted by Year No Pol',{'x':'year','y':'Cost (Bil $)','series':'cost_cat','explode':'scenario','chart_type':'Bar', 'bar_width':'1.75', 'filter': {'cost_cat':{'exclude':costs_pol_inv}}}),
        )),
        }
    ),

    ('Sys Cost Bulk EW (Bil $)',
        {'sources': [
            {'name': 'sc', 'file': 'systemcost_bulk_ew.csv', 'columns': ['cost_cat', 'year', 'Cost (Bil $)']},
        ],
        'index': ['cost_cat', 'year'],
        'preprocess': [
            {'func': pre_systemcost, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Total Discounted',{'x':'scenario','y':'Discounted Cost (Bil $)','series':'cost_cat','chart_type':'Bar', 'filter': {'cost_cat':{'exclude':costs_orig_inv}}}),
            ('Total Discounted No Pol',{'x':'scenario','y':'Discounted Cost (Bil $)','series':'cost_cat','chart_type':'Bar', 'filter': {'cost_cat':{'exclude':costs_pol_inv}}}),
            ('2018-end Discounted',{'x':'scenario','y':'Discounted Cost (Bil $)','series':'cost_cat','chart_type':'Bar', 'filter': {'cost_cat':{'exclude':costs_orig_inv}, 'year': {'start':2018}}}),
            ('2018-end Discounted No Pol',{'x':'scenario','y':'Discounted Cost (Bil $)','series':'cost_cat','chart_type':'Bar', 'filter': {'cost_cat':{'exclude':costs_pol_inv}, 'year': {'start':2018}}}),
            ('Discounted by Year',{'x':'year','y':'Discounted Cost (Bil $)','series':'cost_cat','explode':'scenario','chart_type':'Bar', 'bar_width':'1.75', 'filter': {'cost_cat':{'exclude':costs_orig_inv}}}),
            ('Discounted by Year No Pol',{'x':'year','y':'Discounted Cost (Bil $)','series':'cost_cat','explode':'scenario','chart_type':'Bar', 'bar_width':'1.75', 'filter': {'cost_cat':{'exclude':costs_pol_inv}}}),
            ('Undiscounted by Year',{'x':'year','y':'Cost (Bil $)','series':'cost_cat','explode':'scenario','chart_type':'Bar', 'bar_width':'1.75', 'filter': {'cost_cat':{'exclude':costs_orig_inv}}}),
            ('Undiscounted by Year No Pol',{'x':'year','y':'Cost (Bil $)','series':'cost_cat','explode':'scenario','chart_type':'Bar', 'bar_width':'1.75', 'filter': {'cost_cat':{'exclude':costs_pol_inv}}}),
        )),
        }
    ),

    ('Value Streams inv only',
        {'sources': [
            {'name': 'vs', 'file': 'valuestreams_chosen.csv', 'columns': ['tech', 'vintage', 'n', 'year', 'var_name', 'con_name', '$']},
            {'name': 'cap', 'file': 'cap_new_icrt.csv', 'columns': ['tech', 'vintage', 'region', 'year', 'MW']},
            {'name': 'gen', 'file': 'gen_icrt.csv', 'columns': ['tech', 'vintage', 'n', 'year', 'MWh']},
            {'name': 'pvf_cap', 'file': 'pvf_capital.csv', 'columns': ['year', 'pvfcap']},
            {'name': 'pvf_onm', 'file': 'pvf_onm.csv', 'columns': ['year', 'pvfonm']},
            {'name': 'cost_scale', 'file': 'cost_scale.csv', 'columns': ['cs']},
        ],
        'preprocess': [
            {'func': pre_val_streams, 'args': {'investment_only':True}},
        ],
        'presets': collections.OrderedDict((
            ('NVOE over time', {'x':'year','y':'Bulk $ Dis','series':'con_adj','explode':'scenario','explode_group':'tech','adv_op':'Ratio', 'adv_col':'con_adj', 'adv_col_base':'MWh', 'chart_type':'Bar', 'plot_width':'600', 'plot_height':'600', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'con_name':{'exclude':['kW']}}}),
            ('NVOE final', {'x':'n','y':'Bulk $ Dis','series':'con_adj','explode':'scenario','explode_group':'tech','adv_op':'Ratio', 'adv_col':'con_adj', 'adv_col_base':'MWh', 'chart_type':'Bar', 'plot_width':'600', 'plot_height':'600', 'sync_axes':'No', 'filter': {'year':'last','con_name':{'exclude':['kW']}}}),
            ('NVOC over time', {'x':'year','y':'Bulk $ Dis','series':'con_adj','explode':'scenario','explode_group':'tech','adv_op':'Ratio', 'adv_col':'con_adj', 'adv_col_base':'kW', 'chart_type':'Bar', 'plot_width':'600', 'plot_height':'600', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'con_name':{'exclude':['MWh']}}}),
            ('NVOC final', {'x':'n','y':'Bulk $ Dis','series':'con_adj','explode':'scenario','explode_group':'tech','adv_op':'Ratio', 'adv_col':'con_adj', 'adv_col_base':'kW', 'chart_type':'Bar', 'plot_width':'600', 'plot_height':'600', 'sync_axes':'No', 'filter': {'year':'last','con_name':{'exclude':['MWh']}}}),
            ('LCOE over time', {'x':'year','y':'Bulk $ Dis','series':'con_adj','explode':'scenario','explode_group':'tech','adv_op':'Ratio', 'adv_col':'con_adj', 'adv_col_base':'MWh', 'chart_type':'Bar', 'plot_width':'600', 'plot_height':'600', 'bar_width':'1.75', 'sync_axes':'No', 'y_scale':'-1', 'filter': {'con_name':coststreams+['MWh']}}),
            ('LCOE final', {'x':'n','y':'Bulk $ Dis','series':'con_adj','explode':'scenario','explode_group':'tech','adv_op':'Ratio', 'adv_col':'con_adj', 'adv_col_base':'MWh', 'chart_type':'Bar', 'plot_width':'600', 'plot_height':'600', 'sync_axes':'No', 'y_scale':'-1', 'filter': {'year':'last','con_name':coststreams+['MWh']}}),
            ('NVOE var-con', {'x':'tech, vintage','y':'Bulk $ Dis','series':'var, con', 'explode': 'scenario', 'adv_op':'Ratio', 'adv_col':'var, con', 'adv_col_base':'MWh, MWh', 'chart_type':'Bar', 'plot_width':'600', 'plot_height':'600', 'sync_axes':'No', 'filter': {'con_name':{'exclude':['kW']}}}),
            ('NVOC var-con', {'x':'tech, vintage','y':'Bulk $ Dis','series':'var, con', 'explode': 'scenario', 'adv_op':'Ratio', 'adv_col':'var, con', 'adv_col_base':'kW, kW', 'chart_type':'Bar', 'plot_width':'600', 'plot_height':'600', 'sync_axes':'No', 'filter': {'con_name':{'exclude':['MWh']}}}),
        )),
        }
    ),

    ('Value Streams chosen',
        {'sources': [
            {'name': 'vs', 'file': 'valuestreams_chosen.csv', 'columns': ['tech', 'vintage', 'n', 'year', 'var_name', 'con_name', '$']},
            {'name': 'cap', 'file': 'cap_new_icrt.csv', 'columns': ['tech', 'vintage', 'region', 'year', 'MW']},
            {'name': 'gen', 'file': 'gen_icrt.csv', 'columns': ['tech', 'vintage', 'n', 'year', 'MWh']},
            {'name': 'pvf_cap', 'file': 'pvf_capital.csv', 'columns': ['year', 'pvfcap']},
            {'name': 'pvf_onm', 'file': 'pvf_onm.csv', 'columns': ['year', 'pvfonm']},
            {'name': 'cost_scale', 'file': 'cost_scale.csv', 'columns': ['cs']},
        ],
        'preprocess': [
            {'func': pre_val_streams, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('NVOE', {'x':'tech, vintage','y':'Bulk $ Dis','series':'con_adj', 'explode': 'scenario', 'adv_op':'Ratio', 'adv_col':'con_adj', 'adv_col_base':'MWh', 'chart_type':'Bar', 'plot_width':'600', 'plot_height':'600', 'filter': {'con_name':{'exclude':['kW']}}}),
            ('NVOC', {'x':'tech, vintage','y':'Bulk $ Dis','series':'con_adj', 'explode': 'scenario', 'adv_op':'Ratio', 'adv_col':'con_adj', 'adv_col_base':'kW', 'chart_type':'Bar', 'plot_width':'600', 'plot_height':'600', 'filter': {'con_name':{'exclude':['MWh']}}}),
            ('NVOE var-con', {'x':'tech, vintage','y':'Bulk $ Dis','series':'var, con', 'explode': 'scenario', 'adv_op':'Ratio', 'adv_col':'var, con', 'adv_col_base':'MWh, MWh', 'chart_type':'Bar', 'plot_width':'600', 'plot_height':'600', 'filter': {'con_name':{'exclude':['kW']}}}),
            ('NVOC var-con', {'x':'tech, vintage','y':'Bulk $ Dis','series':'var, con', 'explode': 'scenario', 'adv_op':'Ratio', 'adv_col':'var, con', 'adv_col_base':'kW, kW', 'chart_type':'Bar', 'plot_width':'600', 'plot_height':'600', 'filter': {'con_name':{'exclude':['MWh']}}}),
        )),
        }
    ),

    ('Value Streams chosen OLD',
        {'file':'valuestreams_chosen.csv',
        'columns': ['tech', 'vintage', 'n', 'year', 'new_old', 'var_name', 'con_name', 'value'],
        'preprocess': [
            {'func': pre_val_streams_old, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('$ by type over time', {'x':'year', 'y':'value', 'series':'con_name', 'explode':'scenario', 'explode_group':'tech', 'chart_type':'Bar', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'new_old':['new']}}),
            ('$ by type final', {'chart_type':'Bar', 'x':'tech', 'y':'value', 'series':'con_name', 'explode':'scenario', 'sync_axes':'No', 'bar_width':r'.9', 'cum_sort':'Descending', 'plot_width':'600', 'plot_height':'600', 'filter': {'new_old':['new'], 'con_name':{'exclude':['mwh', 'kw']}, 'year':'last', }}),

            ('$/kW by type over time', {'x':'year', 'y':'value', 'series':'con_name', 'explode':'scenario', 'explode_group':'tech', 'adv_op':'Ratio', 'adv_col':'con_name', 'adv_col_base':'kw', 'chart_type':'Bar', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'con_name':{'exclude':['mwh']}, 'new_old':['new']}}),
            ('$/kW by type final', {'chart_type':'Bar', 'x':'tech', 'y':'value', 'series':'con_name', 'explode':'scenario', 'adv_op':'Ratio', 'adv_col':'con_name', 'adv_col_base':'kw', 'sync_axes':'No', 'bar_width':r'.9', 'cum_sort':'Descending', 'plot_width':'600', 'plot_height':'600', 'filter': {'new_old':['new'], 'con_name':{'exclude':['mwh']}, 'year':'last', }}),

            ('$/MWh by type over time', {'x':'year', 'y':'value', 'series':'con_name', 'explode':'scenario', 'explode_group':'tech', 'adv_op':'Ratio', 'adv_col':'con_name', 'adv_col_base':'mwh', 'chart_type':'Bar', 'bar_width':'1.75', 'sync_axes':'No', 'filter': {'con_name':{'exclude':['kw']}, 'new_old':['new']}}),
            ('$/MWh by type final', {'chart_type':'Bar', 'x':'tech', 'y':'value', 'series':'con_name', 'explode':'scenario', 'adv_op':'Ratio', 'adv_col':'con_name', 'adv_col_base':'mwh', 'sync_axes':'No', 'bar_width':r'.9', 'cum_sort':'Descending', 'plot_width':'600', 'plot_height':'600', 'filter': {'new_old':['new'], 'con_name':{'exclude':['kw']}, 'year':'last', }}),
        )),
        }
    ),

    ('Reduced Cost ($/kW)',
        {'file':'reduced_cost.csv',
        'columns': ['tech', 'vintage', 'region', 'year','bin','variable','$/kW'],
        'preprocess': [
            {'func': pre_reduced_cost, 'args': {}},
            {'func': map_i_to_n, 'args': {}},
            {'func': apply_inflation, 'args': {'column': '$/kW'}},
        ],
        'presets': collections.OrderedDict((
            ('Final supply curves', {'chart_type':'Dot', 'x':'irbv', 'y':'$/kW', 'explode':'scenario','explode_group':'tech', 'sync_axes':'No', 'cum_sort': 'Ascending', 'plot_width':'600', 'plot_height':'600', 'filter': {'year':'last', }}),
            ('Final supply curves p1', {'chart_type':'Dot', 'x':'irbv', 'y':'$/kW', 'explode':'scenario','explode_group':'tech', 'sync_axes':'No', 'cum_sort': 'Ascending', 'plot_width':'600', 'plot_height':'600', 'filter': {'year':'last', 'n':['p1']}}),
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

    ('LCOE cf_act ($/MWh)',
        {'sources': [
            {'name': 'lcoe', 'file': 'lcoe_cf_act.csv', 'columns': ['tech', 'vintage', 'region', 'year', 'bin','$/MWh']},
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

    ('Losses (TWh)',
        {'file':'losses_ann.csv',
        'columns': ['type', 'year', 'Amount (TWh)'],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': .000001, 'column':'Amount (TWh)'}},
        ],
        'index': ['type', 'year'],
        'presets': collections.OrderedDict((
            ('Total Losses Over Time',{'x':'year', 'y':'Amount (TWh)', 'series':'scenario', 'chart_type':'Line', 'filter': {'type':{'exclude':['load']} }}),
            ('Losses by Type Over Time',{'x':'year', 'y':'Amount (TWh)', 'series':'scenario', 'explode':'type', 'chart_type':'Line', 'filter': {'type':{'exclude':['load']} }}),
            ('Fractional Losses by Type Over Time',{'x':'year', 'y':'Amount (TWh)', 'series':'scenario', 'explode':'type', 'chart_type':'Line', 'adv_op':'Ratio', 'adv_col':'type', 'adv_col_base':'load'}),
        )),
        }
    ),

    ('Curtailment Rate',
        {'file':'curt_rate.csv',
        'columns': ['year', 'Curt Rate'],
        'index': ['year'],
        'presets': collections.OrderedDict((
            ('Curt Rate Over Time',{'x':'year', 'y':'Curt Rate', 'series':'scenario', 'chart_type':'Line'}),
        )),
        }
    ),

    ('Curtailment Rate icrt (Realized)',
        {'sources': [
            {'name': 'gen', 'file': 'gen_icrt.csv', 'columns': ['tech', 'vintage', 'n', 'year','MWh']},
            {'name': 'gen_uncurt', 'file': 'gen_icrt_uncurt.csv', 'columns': ['tech', 'vintage', 'n', 'year','MWh uncurt']},
            {'name': 'load', 'file': 'load_rt.csv', 'columns': ['n', 'year','MWh load']},
        ],
        'preprocess': [
            {'func': pre_curt, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Curt Rate Boxplot',{'chart_type':'Dot', 'x':'year', 'y':'Curt Rate', 'y_agg':'None', 'range':'Boxplot', 'explode':'tech', 'explode_group':'scenario', 'sync_axes':'No', 'circle_size':r'3', 'bar_width':r'1.75', }),
            ('Curt Rate weighted ave',{'chart_type':'Line', 'x':'year', 'y':'Curt Rate', 'y_agg':'Weighted Ave', 'y_weight':'MWh uncurt', 'explode':'tech', 'series':'scenario', 'sync_axes':'No', }),
            ('Curt Rate weighted ave vs penetration',{'chart_type':'Line', 'x':'VRE penetration nat', 'y':'Curt Rate', 'y_agg':'Weighted Ave', 'y_weight':'MWh uncurt', 'explode':'tech', 'series':'scenario', 'sync_axes':'No', }),
            ('VRE penetration',{'chart_type':'Line', 'x':'year', 'y':'VRE penetration nat', 'y_agg':'Ave','series':'scenario', 'sync_axes':'No', }),
        )),
        }
    ),

    ('New Tech Curtailment Frac (Caused)',
        {'sources': [
            {'name': 'gen_uncurt', 'file': 'gen_new_uncurt.csv', 'columns': ['tech', 'rr', 'timeslice', 'year', 'MWh uncurt']},
            {'name': 'curt_rate', 'file': 'curt_new.csv', 'columns': ['tech', 'rr', 'timeslice', 'year', 'Curt Rate']},
        ],
        'preprocess': [
            {'func': pre_curt_new, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('Curt Rate Boxplot',{'chart_type':'Dot', 'x':'year', 'y':'Curt Rate', 'y_agg':'None', 'range':'Boxplot', 'explode':'tech', 'explode_group':'scenario', 'sync_axes':'No', 'circle_size':r'3', 'bar_width':r'1.75', }),
            ('Curt Rate weighted ave',{'chart_type':'Line', 'x':'year', 'y':'Curt Rate', 'y_agg':'Weighted Ave', 'y_weight':'MWh uncurt', 'explode':'tech', 'series':'scenario', 'sync_axes':'No', }),
        )),
        }
    ),

    ('New Tech Capacity Credit',
        {'sources': [
            {'name': 'cap', 'file': 'cap_new_cc.csv', 'columns': ['tech', 'rr', 'season', 'year', 'MW']},
            {'name': 'cc', 'file': 'cc_new.csv', 'columns': ['tech', 'rr', 'season', 'year', 'CC Rate']},
        ],
        'preprocess': [
            {'func': pre_cc_new, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('CC Rate Boxplot',{'chart_type':'Dot', 'x':'year', 'y':'CC Rate', 'y_agg':'None', 'range':'Boxplot', 'explode':'season', 'explode_group':'tech', 'series':'scenario', 'sync_axes':'No', 'circle_size':r'3', 'bar_width':r'1.75', }),
            ('CC Rate weighted ave',{'chart_type':'Line', 'x':'year', 'y':'CC Rate', 'y_agg':'Weighted Ave', 'y_weight':'MW', 'explode':'season', 'explode_group':'tech', 'series':'scenario', 'sync_axes':'No', }),
        )),
        }
    ),

    ('Capacity Factor icrt',
        {'sources': [
            {'name': 'gen', 'file': 'gen_icrt.csv', 'columns': ['tech', 'vintage', 'n', 'year','MWh']},
            {'name': 'cap', 'file': 'cap_icrt.csv', 'columns': ['tech', 'vintage', 'n', 'year','MW']},
        ],
        'preprocess': [
            {'func': pre_cf, 'args': {}},
        ],
        'presets': collections.OrderedDict((
            ('CF Boxplot',{'chart_type':'Dot', 'x':'year', 'y':'CF', 'y_agg':'None', 'range':'Boxplot', 'explode':'tech', 'explode_group':'scenario', 'y_min':'0','y_max':'1', 'circle_size':r'3', 'bar_width':r'1.75', }),
            ('CF weighted ave',{'chart_type':'Line', 'x':'year', 'y':'CF', 'y_agg':'Weighted Ave', 'y_weight':'MW', 'explode':'tech', 'series':'scenario', 'y_min':'0','y_max':'1', }),
        )),
        }
    ),

    ('Transmission (GW-mi)',
        {'file':'tran_mi_out.csv',
        'columns': ['year', 'type', 'Amount (GW-mi)'],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': .001, 'column':'Amount (GW-mi)'}},
        ],
        'index': ['type', 'year'],
        'presets': collections.OrderedDict((
            ('Transmission Capacity',{'x':'year', 'y':'Amount (GW-mi)', 'series':'scenario', 'explode':'type', 'chart_type':'Line'}),
        )),
        }
    ),

    ('Transmission Capacity Network (GW)',
        {'file':'tran_out.csv',
        'columns': ['n_out', 'n_in', 'year', 'type', 'Amount (GW)'],
        'preprocess': [
            {'func': add_joint_locations_col, 'args': {'col1':'n_out','col2':'n_in','new':'n-n'}},
            {'func': scale_column, 'args': {'scale_factor': .001, 'column':'Amount (GW)'}},
        ],
        'index': ['n-n', 'year', 'type'],
        'presets': collections.OrderedDict((
            ('Map Final', {'x':'n-n', 'y':'Amount (GW)', 'series':'scenario', 'explode':'year', 'chart_type':'Map', 'filter': {'year': 'last'}}),
            ('Map Final AC/DC', {'x':'n-n', 'y':'Amount (GW)', 'series':'scenario', 'explode':'type', 'explode_group':'year', 'chart_type':'Map', 'filter': {'year': 'last'}}),
            ('Map minus 2018', {'x':'n-n', 'y':'Amount (GW)', 'series':'scenario', 'explode':'year', 'chart_type':'Map', 'adv_op':'Difference', 'adv_col':'year', 'adv_col_base':'2018', 'filter': {'year': ['2018','2050']}}),
        )),
        }
    ),

    ('RE Generation Price ($/MWh)',
        {'file':'RE_gen_price_nat.csv',
        'columns': ['year', 'Price ($/MWh)'],
        'preprocess': [
            {'func': apply_inflation, 'args': {'column':'Price ($/MWh)'}},
        ],
        #'index': ['year'],
        'presets': collections.OrderedDict((
            ('Scenario Lines Over Time',{'x':'year', 'y':'Price ($/MWh)', 'series':'scenario', 'chart_type':'Line'}),
        )),
        }
    ),

    ('RE Capacity Price ($/kW-yr)',
        {'file':'RE_cap_price_nat.csv',
        'columns': ['season', 'year', 'Price ($/kW-yr)'],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': .001, 'column':'Price ($/kW-yr)'}},
            {'func': apply_inflation, 'args': {'column':'Price ($/kW-yr)'}},
        ],
        'index': ['season', 'year'],
        'presets': collections.OrderedDict((
            ('Seasonal RE Capacity Price Over Time',{'x':'year', 'y':'Price ($/kW-yr)', 'series':'scenario', 'explode':'season', 'chart_type':'Line'}),
            ('Total RE Capacity Price Over Time',{'x':'year', 'y':'Price ($/kW-yr)', 'series':'scenario', 'chart_type':'Line'}),
        )),
        }
    ),

    ('RE Capacity Price BA ($/kW-yr)',
        {'file':'RE_cap_price_r.csv',
        'columns': ['n', 'season', 'year', 'Price ($/kW-yr)'],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': .001, 'column':'Price ($/kW-yr)'}},
            {'func': apply_inflation, 'args': {'column':'Price ($/kW-yr)'}},
        ],
        'index': ['n', 'season', 'year'],
        'presets': collections.OrderedDict((
            ('RE Cap Price by BA',{'x':'n', 'y':'Price ($/kW-yr)', 'explode':'scenario', 'explode_group':'season', 'chart_type':'Map', 'filter': {'year':'last'}}),
        )),
        }
    ),

    ('CO2 Price exxon ($/tonne)',
        {'file':'co2_price.csv',
        'columns': ['year', '$/tonne'],
        'preprocess': [
            {'func': apply_inflation, 'args': {'column':'$/tonne'}},
        ],
        'index': ['year'],
        'presets': collections.OrderedDict((
            ('CO2 price over time',{'chart_type':'Line', 'x':'year', 'y':'$/tonne', 'series':'scenario', }),
        )),
        }
    ),

    ('Error Check',
        {'file':'error_check.csv',
        'columns': ['type', 'Value'],
        'presets': collections.OrderedDict((
            ('Errors',{'x':'type', 'y':'Value', 'explode':'scenario', 'chart_type':'Bar'}),
        )),
        }
    ),

    ('Capacity Iteration (GW)',
        {'file':'cap_iter.csv',
        'columns': ['tech', 'vintage', 'rr', 'year', 'iter', 'Capacity (GW)'],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': .001, 'column':'Capacity (GW)'}},
        ],
        'presets': collections.OrderedDict((
            ('Stacked Area',{'x':'year', 'y':'Capacity (GW)', 'series':'tech', 'explode':'iter', 'explode_group':'scenario', 'chart_type':'Area'}),
            ('Stacked Bars',{'x':'year', 'y':'Capacity (GW)', 'series':'tech', 'explode':'iter', 'explode_group':'scenario', 'chart_type':'Bar', 'bar_width':'1.75'}),
            ('Explode By Tech',{'x':'year', 'y':'Capacity (GW)', 'series':'iter', 'explode':'tech', 'explode_group':'scenario', 'chart_type':'Line'}),
            ('PCA Map Final by Tech',{'x':'n', 'y':'Capacity (GW)', 'series':'iter', 'explode':'tech', 'explode_group':'scenario', 'chart_type':'Map', 'filter': {'year':'last'}}),
            ('State Map Final by Tech',{'x':'st', 'y':'Capacity (GW)', 'series':'iter', 'explode':'tech', 'explode_group':'scenario', 'chart_type':'Map', 'filter': {'year':'last'}}),
        )),
        }
    ),

    ('Generation Iteration (TWh)',
        {'file':'gen_iter.csv',
        'columns': ['tech', 'vintage', 'rr', 'year', 'iter', 'Gen (TWh)'],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': 1e-6, 'column':'Gen (TWh)'}},
        ],
        'presets': collections.OrderedDict((
            ('Stacked Area',{'x':'year', 'y':'Gen (TWh)', 'series':'tech', 'explode':'iter', 'explode_group':'scenario', 'chart_type':'Area'}),
            ('Stacked Bars',{'x':'year', 'y':'Gen (TWh)', 'series':'tech', 'explode':'iter', 'explode_group':'scenario', 'chart_type':'Bar', 'bar_width':'1.75'}),
            ('Explode By Tech',{'x':'year', 'y':'Gen (TWh)', 'series':'iter', 'explode':'tech', 'explode_group':'scenario', 'chart_type':'Line'}),
            ('PCA Map Final by Tech',{'x':'n', 'y':'Gen (TWh)', 'series':'iter', 'explode':'tech', 'explode_group':'scenario', 'chart_type':'Map', 'filter': {'year':'last'}}),
            ('State Map Final by Tech',{'x':'st', 'y':'Gen (TWh)', 'series':'iter', 'explode':'tech', 'explode_group':'scenario', 'chart_type':'Map', 'filter': {'year':'last'}}),
        )),
        }
    ),

    ('Firm Capacity Iteration (GW)',
        {'file':'cap_firm_iter.csv',
        'columns': ['tech', 'vintage', 'rr', 'season', 'year', 'iter', 'Capacity (GW)'],
        'preprocess': [
            {'func': scale_column, 'args': {'scale_factor': .001, 'column':'Capacity (GW)'}},
        ],
        'presets': collections.OrderedDict((
            ('Stacked Bars',{'x':'year', 'y':'Capacity (GW)', 'series':'tech', 'explode':'iter', 'explode_group':'scenario', 'chart_type':'Bar', 'bar_width':'1.75'}),
            ('Explode By Tech',{'x':'year', 'y':'Capacity (GW)', 'series':'iter', 'explode':'tech', 'explode_group':'scenario', 'chart_type':'Line'}),
        )),
        }
    ),

    ('Capacity Credit Iteration (GW)',
        {'sources': [
            {'name': 'cap_firm', 'file': 'cap_firm_iter.csv', 'columns': ['tech', 'vintage', 'rr', 'season', 'year', 'iter', 'GW']},
            {'name': 'cap', 'file': 'cap_iter.csv', 'columns': ['tech', 'vintage', 'rr', 'year', 'iter', 'GW']},
        ],
        'preprocess': [
            {'func': pre_cc_iter, 'args': {}},
            {'func': scale_column, 'args': {'scale_factor': .001, 'column':'GW'}},
        ],
        'presets': collections.OrderedDict((
            ('Stacked Bars',{'x':'year', 'y':'GW', 'series':'tech', 'explode':'iter', 'explode_group':'type', 'chart_type':'Bar', 'adv_op':'Ratio', 'adv_col':'type', 'adv_col_base':'cap'}),
            ('Explode By Tech',{'x':'year', 'y':'GW', 'series':'iter', 'explode':'tech', 'explode_group':'type', 'chart_type':'Line', 'adv_op':'Ratio', 'adv_col':'type', 'adv_col_base':'cap'}),
        )),
        }
    ),

    ('Curtailment Iteration (TWh)',
        {'sources': [
            {'name': 'curt', 'file': 'curt_tot_iter.csv', 'columns': ['tech', 'vintage', 'rr', 'year', 'iter', 'TWh']},
            {'name': 'gen_uncurt', 'file': 'gen_iter.csv', 'columns': ['tech', 'vintage', 'rr', 'year', 'iter', 'TWh']},
        ],
        'preprocess': [
            {'func': pre_curt_iter, 'args': {}},
            {'func': scale_column, 'args': {'scale_factor': 1e-6, 'column':'TWh'}},
        ],
        'presets': collections.OrderedDict((
            ('Explode By Tech',{'x':'year', 'y':'TWh', 'series':'iter', 'explode':'tech', 'explode_group':'type', 'chart_type':'Line', 'adv_op':'Ratio', 'adv_col':'type', 'adv_col_base':'gen', }),
        )),
        }
    ),
))
