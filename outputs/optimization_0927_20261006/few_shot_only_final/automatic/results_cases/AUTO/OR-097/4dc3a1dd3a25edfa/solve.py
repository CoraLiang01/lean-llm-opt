import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
optchar_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
optchar_df = pd.read_csv(optchar_path, sep=',', dtype=str, keep_default_na=False)
assetref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
assetref_df = pd.read_csv(assetref_path, sep=',', dtype=str, keep_default_na=False)
option_ids = list(optchar_df['Option'])
asset_cols = [col for col in assetref_df.columns if re.match('Asset_\\d+', col)]
asset_ids = [col for col in asset_cols]
greek_names = ['Delta', 'Gamma', 'Vega']
greek_initial = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
greek_tol = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}

def col_to_float_dict(df, colname):
    return {row['Option']: float(row[colname]) for (_, row) in df.iterrows()}
cost = col_to_float_dict(optchar_df, 'Cost')
delta = col_to_float_dict(optchar_df, 'Delta')
gamma = col_to_float_dict(optchar_df, 'Gamma')
vega = col_to_float_dict(optchar_df, 'Vega')
maxlong = col_to_float_dict(optchar_df, 'MaxLong')
maxshort = col_to_float_dict(optchar_df, 'MaxShort')
greek_per_option = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
if 'Unnamed: 0' in assetref_df.columns:
    assetref_df = assetref_df.set_index('Unnamed: 0')
else:
    raise ValueError("Option_AssetReferenceMatrix.csv must have 'Unnamed: 0' as the option identifier column.")
missing_opts = set(option_ids) - set(assetref_df.index)
if missing_opts:
    raise ValueError(f'Options missing in Option_AssetReferenceMatrix.csv: {missing_opts}')
A = {}
for opt in option_ids:
    A[opt] = {}
    for asset in asset_ids:
        val = assetref_df.loc[opt, asset]
        try:
            A[opt][asset] = int(val)
        except Exception:
            raise ValueError(f"Non-integer asset reference for option {opt}, asset {asset}: '{val}'")
m = gp.Model('OptionHedging')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb={opt: maxshort[opt] for opt in option_ids}, ub={opt: maxlong[opt] for opt in option_ids}, name='')
z_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, name='')
m.setObjective(gp.quicksum((cost[opt] * z_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
for opt in option_ids:
    m.addConstr(z_vars[opt] >= x_vars[opt], name=f'abs_pos_{opt}')
    m.addConstr(z_vars[opt] >= -x_vars[opt], name=f'abs_neg_{opt}')
for greek in greek_names:
    greek_vec = greek_per_option[greek]
    expr = gp.LinExpr()
    for opt in option_ids:
        for asset in asset_ids:
            if A[opt][asset] != 0:
                expr.addTerms(greek_vec[opt], x_vars[opt])
    m.addConstr(greek_initial[greek] + expr <= greek_tol[greek], name=f'{greek}_plus')
    m.addConstr(greek_initial[greek] + expr >= -greek_tol[greek], name=f'{greek}_minus')
m.optimize()