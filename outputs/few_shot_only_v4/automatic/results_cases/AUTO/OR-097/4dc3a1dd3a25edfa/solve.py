import gurobipy as gp
import pandas as pd
import numpy as np
import re
option_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
option_assetref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
option_char_df = pd.read_csv(option_char_path, sep=',', dtype=str, keep_default_na=False)
option_assetref_df = pd.read_csv(option_assetref_path, sep=',', dtype=str, keep_default_na=False)
if 'Option' not in option_char_df.columns:
    raise KeyError("OptionCharacteristics.csv must have an 'Option' column.")
option_ids = option_char_df['Option'].tolist()
if len(option_ids) != 120:
    raise ValueError(f'Expected 120 options, found {len(option_ids)} in OptionCharacteristics.csv.')
assetref_cols = list(option_assetref_df.columns)
if assetref_cols[0].strip().lower() in {'option', 'optionid', 'id'}:
    asset_ids = assetref_cols[1:]
    option_assetref_df = option_assetref_df.set_index(assetref_cols[0])
else:
    asset_ids = assetref_cols
if len(asset_ids) != 6:
    raise ValueError(f'Expected 6 assets, found {len(asset_ids)} in Option_AssetReferenceMatrix.csv.')
assetref_option_ids = [str(idx) for idx in option_assetref_df.index]
if set(option_ids) != set(assetref_option_ids):
    raise ValueError('Mismatch between options in OptionCharacteristics.csv and Option_AssetReferenceMatrix.csv.')
option_assetref_df = option_assetref_df.loc[option_ids, asset_ids]
A = {}
for i in option_ids:
    for j in asset_ids:
        val = option_assetref_df.loc[i, j].strip()
        if val not in {'0', '1'}:
            raise ValueError(f'Invalid A[{i},{j}] value: {val}')
        A[i, j] = int(val)

def parse_numeric_column(df, col, ids, allow_negative=False):
    vals = {}
    for idx, row in df.iterrows():
        key = row['Option']
        val = row[col].strip()
        try:
            fval = float(val)
        except Exception:
            raise ValueError(f"Invalid numeric value in column '{col}' for Option '{key}': '{val}'")
        if not allow_negative and fval < 0:
            raise ValueError(f"Negative value in column '{col}' for Option '{key}': '{val}'")
        vals[key] = fval
    if set(vals.keys()) != set(ids):
        raise ValueError(f"Missing or extra keys in column '{col}'.")
    return vals
cost = parse_numeric_column(option_char_df, 'Cost', option_ids)
delta = parse_numeric_column(option_char_df, 'Delta', option_ids, allow_negative=True)
gamma = parse_numeric_column(option_char_df, 'Gamma', option_ids, allow_negative=True)
vega = parse_numeric_column(option_char_df, 'Vega', option_ids, allow_negative=True)
maxlong = parse_numeric_column(option_char_df, 'MaxLong', option_ids, allow_negative=False)
maxshort = parse_numeric_column(option_char_df, 'MaxShort', option_ids, allow_negative=True)
greek_names = ['Delta', 'Gamma', 'Vega']
greek_initial = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
greek_tolerance = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
greek_per_option = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
m = gp.Model('OptionHedging')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb={i: maxshort[i] for i in option_ids}, ub={i: maxlong[i] for i in option_ids}, name='')
z_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost[i] * z_vars[i] for i in option_ids)), gp.GRB.MINIMIZE)
for i in option_ids:
    m.addConstr(z_vars[i] >= x_vars[i], name=f'abs_pos_{i}')
    m.addConstr(z_vars[i] >= -x_vars[i], name=f'abs_neg_{i}')
for greek in greek_names:
    expr = gp.LinExpr()
    for i in option_ids:
        for j in asset_ids:
            if A[i, j] == 1:
                expr.addTerms(greek_per_option[greek][i], x_vars[i])
    m.addConstr(greek_initial[greek] + expr <= greek_tolerance[greek], name=f'{greek}_upper')
    m.addConstr(greek_initial[greek] + expr >= -greek_tolerance[greek], name=f'{greek}_lower')
m.optimize()