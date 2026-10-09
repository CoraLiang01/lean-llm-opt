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
if len(option_ids) != 120:
    raise ValueError(f'Expected 120 options, got {len(option_ids)}')
asset_cols = [col for col in assetref_df.columns if re.match('Asset_\\d+', col)]
asset_ids = [col for col in asset_cols]
if len(asset_ids) != 6:
    raise ValueError(f'Expected 6 assets, got {len(asset_ids)}')

def col_to_float(series, name):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f'Could not convert column {name} to float: {e}')
cost_param = dict(zip(option_ids, col_to_float(optchar_df['Cost'], 'Cost')))
delta_param = dict(zip(option_ids, col_to_float(optchar_df['Delta'], 'Delta')))
gamma_param = dict(zip(option_ids, col_to_float(optchar_df['Gamma'], 'Gamma')))
vega_param = dict(zip(option_ids, col_to_float(optchar_df['Vega'], 'Vega')))
maxlong_param = dict(zip(option_ids, optchar_df['MaxLong'].astype(int)))
maxshort_param = dict(zip(option_ids, optchar_df['MaxShort'].astype(int)))
if not all(assetref_df['Unnamed: 0'].values == np.array(option_ids)):
    assetref_df = assetref_df.set_index('Unnamed: 0').reindex(option_ids)
    if assetref_df.isnull().any().any():
        raise ValueError('Option identifiers in Option_AssetReferenceMatrix.csv do not match OptionCharacteristics.csv')
else:
    assetref_df = assetref_df.set_index('Unnamed: 0')
A = {}
for i in option_ids:
    for j in asset_ids:
        val = assetref_df.loc[i, j]
        try:
            A[i, j] = int(val)
        except Exception:
            raise ValueError(f'Non-integer value in asset reference matrix at option {i}, asset {j}: {val}')
greeks = ['Delta', 'Gamma', 'Vega']
initial_exposure = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
tolerance = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
greek_param = {'Delta': delta_param, 'Gamma': gamma_param, 'Vega': vega_param}
m = gp.Model('OptionHedging')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=[maxshort_param[i] for i in option_ids], ub=[maxlong_param[i] for i in option_ids], name='')
z_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
for i in option_ids:
    m.addConstr(z_vars[i] >= x_vars[i], name=f'abs_pos_{i}')
    m.addConstr(z_vars[i] >= -x_vars[i], name=f'abs_neg_{i}')
for G in greeks:
    expr = gp.LinExpr()
    for i in option_ids:
        for j in asset_ids:
            coeff = greek_param[G][i] * A[i, j]
            if abs(coeff) > 1e-12:
                expr.addTerms(coeff, x_vars[i])
    expr_total = initial_exposure[G] + expr
    m.addConstr(expr_total <= tolerance[G], name=f'{G}_upper')
    m.addConstr(expr_total >= -tolerance[G], name=f'{G}_lower')
m.setObjective(gp.quicksum((cost_param[i] * z_vars[i] for i in option_ids)), gp.GRB.MINIMIZE)
m.optimize()