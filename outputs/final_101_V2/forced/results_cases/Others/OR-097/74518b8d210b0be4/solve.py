import gurobipy as gp
import pandas as pd
import numpy as np
import re
opt_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
opt_df = pd.read_csv(opt_char_path, sep=',')
asset_ref_df = pd.read_csv(asset_ref_path, sep=',')
options = [str(opt).strip() for opt in opt_df['Option']]
if len(options) != 120:
    raise ValueError(f'Expected 120 options, got {len(options)}')
asset_cols = [col for col in asset_ref_df.columns if re.match('Asset_\\d+', col)]
assets = asset_cols
if len(assets) != 6:
    raise ValueError(f'Expected 6 assets, got {len(assets)}')
asset_ref_df['Option'] = asset_ref_df['Unnamed: 0'].astype(str).str.strip()
asset_ref_df = asset_ref_df.set_index('Option')
missing_opts = set(options) - set(asset_ref_df.index)
if missing_opts:
    raise ValueError(f'Options missing in asset reference matrix: {missing_opts}')
cost = dict(zip(options, opt_df['Cost']))
delta = dict(zip(options, opt_df['Delta']))
gamma = dict(zip(options, opt_df['Gamma']))
vega = dict(zip(options, opt_df['Vega']))
maxlong = dict(zip(options, opt_df['MaxLong']))
maxshort = dict(zip(options, opt_df['MaxShort']))
A = {}
for opt in options:
    A[opt] = {}
    for asset in assets:
        A[opt][asset] = int(asset_ref_df.loc[opt, asset])
initial_greek = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
tolerance_greek = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
m = gp.Model('OptionHedging')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb={opt: maxshort[opt] for opt in options}, ub={opt: maxlong[opt] for opt in options}, name='')
z = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
for opt in options:
    m.addConstr(z[opt] >= x[opt], name=f'abs1_{opt}')
    m.addConstr(z[opt] >= -x[opt], name=f'abs2_{opt}')
for greek, greek_vec in [('Delta', delta), ('Gamma', gamma), ('Vega', vega)]:
    expr = gp.LinExpr()
    for opt in options:
        for asset in assets:
            if A[opt][asset] != 0:
                expr.addTerms(greek_vec[opt] * A[opt][asset], x[opt])
    m.addConstr(initial_greek[greek] + expr <= tolerance_greek[greek], name=f'{greek}_upper')
    m.addConstr(initial_greek[greek] + expr >= -tolerance_greek[greek], name=f'{greek}_lower')
m.setObjective(gp.quicksum((cost[opt] * z[opt] for opt in options)), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total hedging cost: {m.objVal:.2f}')
    print('--- Option Positions ---')
    for opt in options:
        xi = x[opt].X
        if abs(xi) > 1e-06:
            print(f'{opt}: {int(round(xi)):+d} contracts (Cost per contract: {cost[opt]})')
    exposures = {}
    for greek, greek_vec in [('Delta', delta), ('Gamma', gamma), ('Vega', vega)]:
        total = initial_greek[greek]
        for opt in options:
            for asset in assets:
                if A[opt][asset] != 0:
                    total += greek_vec[opt] * A[opt][asset] * x[opt].X
        exposures[greek] = total
    print('--- Final Exposures After Hedging ---')
    for greek in ['Delta', 'Gamma', 'Vega']:
        print(f'{greek}: {exposures[greek]:+.5f} (Tolerance: ±{tolerance_greek[greek]:.5f})')
else:
    print(f'No optimal solution found. Status: {m.status}')