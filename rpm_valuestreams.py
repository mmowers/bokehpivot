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