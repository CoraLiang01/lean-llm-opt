import gurobipy as gp
import pandas as pd
import numpy as np
import re
optchar_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
optchar_df = pd.read_csv(optchar_path, sep=',', dtype=str, keep_default_na=False)
optchar_df = optchar_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
optasset_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
optasset_df = pd.read_csv(optasset_path, sep=',', dtype=str, keep_default_na=False)
optasset_df = optasset_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
option_ids = [oid.strip() for oid in optchar_df['Option']]
n_options = len(option_ids)
asset_cols = [col for col in optasset_df.columns if re.match('Asset_\\d+', col)]
asset_ids = asset_cols
n_assets = len(asset_ids)
optchar_df = optchar_df.set_index('Option', drop=False)
for col in ['Cost', 'Delta', 'Gamma', 'Vega', 'MaxLong', 'MaxShort']:
    optchar_df[col] = pd.to_numeric(optchar_df[col], errors='raise')
optasset_df = optasset_df.set_index('Unnamed: 0', drop=False)
for col in asset_cols:
    optasset_df[col] = pd.to_numeric(optasset_df[col], errors='raise')
if not set(option_ids).issubset(set(optasset_df.index)):
    missing = set(option_ids) - set(optasset_df.index)
    raise ValueError(f'Options missing in Option_AssetReferenceMatrix: {missing}')
if not set(option_ids).issubset(set(optchar_df.index)):
    missing = set(option_ids) - set(optchar_df.index)
    raise ValueError(f'Options missing in OptionCharacteristics: {missing}')
cost = {oid: float(optchar_df.loc[oid, 'Cost']) for oid in option_ids}
delta = {oid: float(optchar_df.loc[oid, 'Delta']) for oid in option_ids}
gamma = {oid: float(optchar_df.loc[oid, 'Gamma']) for oid in option_ids}
vega = {oid: float(optchar_df.loc[oid, 'Vega']) for oid in option_ids}
maxlong = {oid: int(optchar_df.loc[oid, 'MaxLong']) for oid in option_ids}
maxshort = {oid: int(optchar_df.loc[oid, 'MaxShort']) for oid in option_ids}
A = {oid: {aid: int(optasset_df.loc[oid, aid]) for aid in asset_ids} for oid in option_ids}
greek_names = ['Delta', 'Gamma', 'Vega']
greek_initial = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
greek_tol = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
greek_coeff = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
m = gp.Model('OptionHedging')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=[maxshort[oid] for oid in option_ids], ub=[maxlong[oid] for oid in option_ids], name='')
z_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost[oid] * z_vars[oid] for oid in option_ids)), gp.GRB.MINIMIZE)
for oid in option_ids:
    m.addConstr(z_vars[oid] >= x_vars[oid], name=f'abs1_{oid}')
    m.addConstr(z_vars[oid] >= -x_vars[oid], name=f'abs2_{oid}')
for greek in greek_names:
    nref_assets = {oid: sum((A[oid][aid] for aid in asset_ids)) for oid in option_ids}
    expr = greek_initial[greek] + gp.quicksum((x_vars[oid] * greek_coeff[greek][oid] * A[oid][aid] for oid in option_ids for aid in asset_ids))
    m.addConstr(expr <= greek_tol[greek], name=f'{greek}_upper')
    m.addConstr(expr >= -greek_tol[greek], name=f'{greek}_lower')
m.optimize()