import gurobipy as gp
import pandas as pd
import numpy as np
import re
optchar_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
optchar_df = pd.read_csv(optchar_path, sep=',', dtype=str, keep_default_na=False)
assetref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
assetref_df = pd.read_csv(assetref_path, sep=',', dtype=str, keep_default_na=False)
option_ids = optchar_df['Option'].tolist()
n_options = len(option_ids)
asset_cols = [col for col in assetref_df.columns if re.match('Asset_\\d+', col)]
asset_ids = asset_cols
n_assets = len(asset_ids)
option_id_to_idx = {oid: idx for idx, oid in enumerate(option_ids)}
assetref_df_indexed = assetref_df.set_index('Unnamed: 0')
assetref_option_ids = assetref_df_indexed.index.tolist()
if set(option_ids) != set(assetref_option_ids):
    raise ValueError('Option IDs in OptionCharacteristics and Option_AssetReferenceMatrix do not match.')
cost = optchar_df.set_index('Option')['Cost'].astype(float).to_dict()
delta = optchar_df.set_index('Option')['Delta'].astype(float).to_dict()
gamma = optchar_df.set_index('Option')['Gamma'].astype(float).to_dict()
vega = optchar_df.set_index('Option')['Vega'].astype(float).to_dict()
maxlong = optchar_df.set_index('Option')['MaxLong'].astype(int).to_dict()
maxshort = optchar_df.set_index('Option')['MaxShort'].astype(int).to_dict()
A = {}
for oid in option_ids:
    for aid in asset_ids:
        A[oid, aid] = int(assetref_df_indexed.loc[oid, aid])
greek_names = ['Delta', 'Gamma', 'Vega']
greek_initial = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
greek_tolerance = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
greek_param = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
m = gp.Model('OptionHedging')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb={oid: maxshort[oid] for oid in option_ids}, ub={oid: maxlong[oid] for oid in option_ids}, name='')
y_ub = {oid: max(abs(maxlong[oid]), abs(maxshort[oid])) for oid in option_ids}
y_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, ub=y_ub, name='')
m.setObjective(gp.quicksum((cost[oid] * y_vars[oid] for oid in option_ids)), gp.GRB.MINIMIZE)
for oid in option_ids:
    m.addConstr(y_vars[oid] >= x_vars[oid], name=f'abs_pos_{oid}')
    m.addConstr(y_vars[oid] >= -x_vars[oid], name=f'abs_neg_{oid}')
for greek in greek_names:
    greek_coeff = greek_param[greek]
    expr = greek_initial[greek] + gp.quicksum((greek_coeff[oid] * A[oid, aid] * x_vars[oid] for oid in option_ids for aid in asset_ids))
    tol = greek_tolerance[greek]
    m.addConstr(expr <= tol, name=f'{greek}_upper')
    m.addConstr(expr >= -tol, name=f'{greek}_lower')
m.optimize()