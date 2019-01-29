ba_list = ['ldwp',  'aeso', 'aps',  'ava',  'bctc', 'bpa',  'cfe',  'epe',  'far_east', 'iid',  'magic_vly',    'nevp', 'nwmt', 'pace_id',  'pace_ut',  'pace_wy',  'pacw', 'pg&e_bay', 'pg&e_vly', 'pgn',  'pnm',  'psc',  'pse',  'sce', 'scl',  'sdge', 'smud', 'spp',  'srp',  'tep',  'tidc', 'tpwr', 'treas_vly', 'wacm', 'walc', 'wauw']
node_list = ['14002_moenkopi',   '14003_navajo', '19011_mead n', '19012_mead s', '19038_mead',   '24041_eldordo',    '24042_eldordo',    '24086_lugo',   '24097_mohave', '24729_inyo',   '26003_adelanto',   '26004_castai1g',   '26005_castai2g',   '26006_castai3g',   '26007_castai4g',   '26008_castai5g',   '26009_castai6g',   '26010_castaic',    '26013_glendal',    '26014_gramercy',   '26016_halldale',   '26023_harb5g', '26025_haynes', '26026_haynes1g',   '26027_haynes2g',   '26032_owens up',   '26033_owensmid',   '26034_owenscon',   '26039_interm1g',   '26040_interm2g',   '26041_intermt',    '26043_intermt',    '26044_marketpl',   '26046_mccullgh',   '26048_mccullgh',   '26051_mead',   '26052_olive',  '26057_pp 1',   '26058_pp 1  g',    '26059_pp 2',   '26060_pp 2  g',    '26061_rinaldi',    '26062_rinaldi',    '26063_river',  '26065_scatergd',   '26066_scatergd',   '26067_scatt3g',    '26068_stjohn', '26069_cntury', '26070_cntury1',    '26071_cntury2',    '26072_cnturyld',   '26073_wlmntn', '26076_fairfax',    '26077_toluca', '26078_toluca', '26079_toluca', '26080_velasco',    '26081_atwater',    '26082_holywd_e',   '26085_holywdld',   '26086_nrthrdge',   '26087_olympc', '26088_olympcld',   '26089_airport',    '26091_harbor', '26093_tarzana',    '26095_tap 1',  '26096_tap 2',  '26097_sylmar1',    '26102_valley', '26103_valley', '26104_victorvl',   '26105_victorvl',   '26106_scatt2g',   '26111_harb2g', '26112_scatt1g',    '26123_crystal',    '26129_owens up',   '26130_owensmid',   '26131_owenscon',   '26132_barrenrd',   '26135_hskllcyn',   '26136_cottonwd',   '26142_valley5g',   '26143_harbct10',   '26144_harbct11',   '26145_harbct12',   '26146_harbct13',   '26147_harbct14',   '26151_haynes8g',   '26152_haynes9g',   '26153_hayns10g',   '26156_hyn1516g',   '26182_holywd_f',   '26905_bcon230',    '26907_bcon18g',    '26934_sodatap',    '27001_ptwtg',  '27031_pt230',  '27204_hltap',  '41311_celilo1',    '41313_celilo3',    '85998_market', '85999_vannuys',    '86000_canoga']
tech_list = ['nuclear','coal','gas-cc','gas-ct','gas-other','ogs','hydro','bio','geo','wind','pv-track','pv-roof','pv-fix','pv-bat','csp-tes','csp-notes','storage']

static_presets = []

#for ba in ba_list:
#Configuration 1 - section=region | x=tech | explode=year 
#    static_presets.append({'name': 'New Value Streams by BA: ' + ba.upper(), 'result': 'Value Streams BA', 'preset': 'x=tech | ex=year', 'config':{'filter':{'ba':[ba],'new_old':['new']}}})
#    static_presets.append({'name': 'Old Value Streams by BA: ' + ba.upper(), 'result': 'Value Streams BA', 'preset': 'x=tech | ex=year', 'config':{'filter':{'ba':[ba],'new_old':['old']}}})	

#Configuration 2 - section=region | x=class; x-group=tech | explode=year 
#    static_presets.append({'name': 'New Value Streams by BA: ' + ba.upper(), 'result': 'Value Streams BA', 'preset': 'x=class | x-gp=tech | ex=year', 'config':{'filter':{'ba':[ba],'new_old':['new']}}})
#    static_presets.append({'name': 'Old Value Streams by BA: ' + ba.upper(), 'result': 'Value Streams BA', 'preset': 'x=class | x-gp=tech | ex=year', 'config':{'filter':{'ba':[ba],'new_old':['old']}}})

#for tech in tech_list:																									
#Configuration 3 - section=tech | x=region | explode=year 
#    static_presets.append({'name': 'New Value Streams by Tech: ' + tech.upper(), 'result': 'Value Streams BA', 'preset': 'x=ba | ex=year', 'config':{'filter':{'tech':[tech],'new_old':['new']}}})
#    static_presets.append({'name': 'Old Value Streams by Tech: ' + tech.upper(), 'result': 'Value Streams BA', 'preset': 'x=ba | ex=year', 'config':{'filter':{'tech':[tech],'new_old':['old']}}})

# ===================================
		
#for node in node_list:
#Configuration 1 - section=region | x=tech | explode=year
#    static_presets.append({'name': 'New Value Streams by Node: ' + node.upper(), 'result': 'Value Streams Node', 'preset': 'x=tech | ex=year', 'config':{'filter':{'node':[node],'new_old':['new']}}})
#    static_presets.append({'name': 'Old Value Streams by Node: ' + node.upper(), 'result': 'Value Streams Node', 'preset': 'x=tech | ex=year', 'config':{'filter':{'node':[node],'new_old':['old']}}})

#Configuration 2 - section=region | x=class; x-group=tech | explode=year 
#    static_presets.append({'name': 'New Value Streams by Node: ' + node.upper(), 'result': 'Value Streams Node', 'preset': 'x=class | x-gp=tech | ex=year', 'config':{'filter':{'node':[node],'new_old':['new']}}})
#    static_presets.append({'name': 'Old Value Streams by Node: ' + node.upper(), 'result': 'Value Streams Node', 'preset': 'x=class | x-gp=tech | ex=year', 'config':{'filter':{'node':[node],'new_old':['old']}}})

for tech in tech_list:																									
#Configuration 3 - section=tech | x=region | explode=year 
    static_presets.append({'name': 'New Value Streams by Tech: ' + tech.upper(), 'result': 'Value Streams Node', 'preset': 'x=node | ex=year', 'config':{'filter':{'tech':[tech],'new_old':['new']}}})
#   static_presets.append({'name': 'Old Value Streams by Tech: ' + tech.upper(), 'result': 'Value Streams Node', 'preset': 'x=node | ex=year', 'config':{'filter':{'tech':[tech],'new_old':['old']}}})
