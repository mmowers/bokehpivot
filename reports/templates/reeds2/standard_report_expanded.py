static_presets = [
    {'name': 'Error Check', 'result': 'Error Check', 'preset': 'Errors'},
    {'name': 'Intertemporal Capacity by Iteration (GW)', 'result': 'Capacity Iteration (GW)', 'preset': 'Explode By Tech'},
    {'name': 'Generation (TWh)', 'result': 'Generation BA (TWh)', 'preset': 'Stacked Bars'},
    {'name': 'Capacity (GW)', 'result': 'Capacity BA (GW)', 'preset': 'Stacked Bars'},
    {'name': 'New Annual Capacity (GW)', 'result': 'New Annual Capacity BA (GW)', 'preset': 'Stacked Bars'},
    {'name': 'Annual Retirements (GW)', 'result': 'Annual Retirements BA (GW)', 'preset': 'Stacked Bars'},
    {'name': 'Final Gen by timeslice (GW)', 'result': 'Gen by timeslice national (GW)', 'preset': 'Stacked Bars Final'},
    {'name': 'Operating Reserves (TW-h)', 'result': 'Operating Reserves (TW-h)', 'preset': 'Stacked Bars'},
    {'name': 'Final OpRes by timeslice (GW)', 'result': 'Operating Reserves by Timeslice National (GW)', 'preset': 'Stacked Bars Final'},
    {'name': 'Firm Capacity (GW)', 'result': 'Firm Capacity (GW)', 'preset': 'Stacked Bars'},
    {'name': 'Curtailment Rate', 'result': 'Average VRE Curtailment', 'preset': 'Curt Rate Over Time'},
    {'name': 'Losses (fraction of load)', 'result': 'Losses (TWh)', 'preset': 'Fractional Losses by Type Over Time'},
    {'name': 'Transmission (GW-mi)', 'result': 'Transmission (GW-mi)', 'preset': 'Transmission Capacity'},
    {'name': 'All-in Electricity Price ($/MWh)', 'result': 'Requirement Prices and Quantities', 'preset': 'All-in Price ($/MWh)'},
    {'name': 'National Energy Price ($/MWh)', 'result': 'Requirement Prices and Quantities', 'preset': 'Energy Price Lines ($/MWh)'},
    {'name': 'National Energy Price by timeslice ($/MWh)', 'result': 'Requirement Prices and Quantities', 'preset': 'Energy Price by Timeslice Final ($/MWh)'},
    {'name': 'National Average Electricity Price ($/MWh)', 'result': 'National Average Electricity Price ($/MWh)', 'preset': 'Average Electricity Price by Year ($/MWh)'},
    {'name': 'National OpRes Price ($/MW-h)', 'result': 'Requirement Prices and Quantities', 'preset': 'OpRes Price Lines ($/MW-h)'},
    {'name': 'National OpRes Price by timeslice ($/MW-h)', 'result': 'Requirement Prices and Quantities', 'preset': 'OpRes Price by Timeslice Final ($/MW-h)'},
    {'name': 'Final Regional Energy Price ($/MWh)', 'result': 'Requirement Prices and Quantities', 'preset': 'Energy Price Final BA Map ($/MWh)'},
    {'name': 'National Seasonal Capacity Price ($/kW-yr)', 'result': 'Requirement Prices and Quantities', 'preset': 'ResMarg Season Price Lines ($/kW-yr)'},
    {'name': 'National Annual Capacity Price ($/kW-yr)', 'result': 'Requirement Prices and Quantities', 'preset': 'ResMarg Price Lines ($/kW-yr)'},
    {'name': 'System Cost (Bil $)', 'result': 'Sys Cost truncated at final year (Bil $)', 'preset': 'Total Discounted'},
    {'name': 'CO2 Emissions National (MMton)', 'result': 'CO2 Emissions National (MMton)', 'preset': 'Scenario Lines Over Time'},
    {'name': 'Final Wind Capacity (GW)', 'result': 'Capacity Resource Region (GW)', 'preset': 'RS Map Final Wind'},
    {'name': 'PV Capacity (GW)', 'result': 'Capacity BA (GW)', 'preset': 'PCA Map Final by Tech','config':{'explode_group':'None','filter':{'tech':['upv','dupv','distpv']}}},
    {'name': 'CSP Capacity (GW)', 'result': 'Capacity BA (GW)', 'preset': 'PCA Map Final by Tech','config':{'explode_group':'None','filter':{'tech':['csp']}}},
    {'name': 'Biopower Capacity (GW)', 'result': 'Capacity BA (GW)', 'preset': 'PCA Map Final by Tech','config':{'explode_group':'None','filter':{'tech':['biopower','lfill-gas']}}},
    {'name': 'Geothermal Capacity (GW)', 'result': 'Capacity BA (GW)', 'preset': 'PCA Map Final by Tech','config':{'explode_group':'None','filter':{'tech':['geothermal']}}},
    {'name': 'Hydro and Canadian Import Capacity (GW)', 'result': 'Capacity BA (GW)', 'preset': 'PCA Map Final by Tech','config':{'explode_group':'None','filter':{'tech':['hydro','Canada']}}},
    {'name': 'Pumped-hydro Capacity (GW)', 'result': 'Capacity BA (GW)', 'preset': 'PCA Map Final by Tech','config':{'explode_group':'None','filter':{'tech':['pumped-hydro']}}},
    {'name': 'Battery Storage Capacity (GW)', 'result': 'Capacity BA (GW)', 'preset': 'PCA Map Final by Tech','config':{'explode_group':'None','filter':{'tech':['battery', 'battery_2', 'battery_4', 'battery_6', 'battery_8', 'battery_10']}}},
]
