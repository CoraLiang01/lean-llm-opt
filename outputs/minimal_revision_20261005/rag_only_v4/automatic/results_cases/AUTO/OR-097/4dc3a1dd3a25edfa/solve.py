import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
opt_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
opt_df = pd.read_csv(opt_char_path, sep=',')
asset_ref_df = pd.read_csv(asset_ref_path, sep=',')
opt_df['Option'] = opt_df['Option'].astype(str).str.strip()
asset_ref_df['Unnamed: 0'] = asset_ref_df['Unnamed: 0'].astype(str).str.strip()
option_ids = list(opt_df['Option'])
asset_ids = [col for col in asset_ref_df.columns if col.startswith('Asset_')]
if set(asset_ref_df['Unnamed: 0']) != set(option_ids):
    raise ValueError('Mismatch between OptionCharacteristics and Option_AssetReferenceMatrix option identifiers.')
option_idx = {oid: i for (i, oid) in enumerate(option_ids)}
asset_idx = {aid: j for (j, aid) in enumerate(asset_ids)}
cost = dict(zip(opt_df['Option'], opt_df['Cost']))
delta = dict(zip(opt_df['Option'], opt_df['Delta']))
gamma = dict(zip(opt_df['Option'], opt_df['Gamma']))
vega = dict(zip(opt_df['Option'], opt_df['Vega']))
maxlong = dict(zip(opt_df['Option'], opt_df['MaxLong']))
maxshort = dict(zip(opt_df['Option'], opt_df['MaxShort']))
A = {}
for (_, row) in asset_ref_df.iterrows():
    oid = row['Unnamed: 0']
    for aid in asset_ids:
        A[oid, aid] = int(row[aid])
GREEKS = ['Delta', 'Gamma', 'Vega']
GREEK_INIT = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
GREEK_TOL = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
GREEK_COEF = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
m = gp.Model('option_hedging')
m.Params.MIPGap = 0.0001
x = m.addVars(option_ids, vtype=GRB.INTEGER, lb=[maxshort[oid] for oid in option_ids], ub=[maxlong[oid] for oid in option_ids], name='')
abs_x = m.addVars(option_ids, vtype=GRB.CONTINUOUS, lb=0.0, name='')
for oid in option_ids:
    m.addConstr(abs_x[oid] >= x[oid], name='')
    m.addConstr(abs_x[oid] >= -x[oid], name='')
m.setObjective(gp.quicksum((cost[oid] * abs_x[oid] for oid in option_ids)), GRB.MINIMIZE)
for greek in GREEKS:
    greek_coef = GREEK_COEF[greek]
    expr = GREEK_INIT[greek] + gp.quicksum((greek_coef[oid] * A[oid, aid] * x[oid] for oid in option_ids for aid in asset_ids))
    tol = GREEK_TOL[greek]
    m.addConstr(expr <= tol, name='')
    m.addConstr(expr >= -tol, name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')