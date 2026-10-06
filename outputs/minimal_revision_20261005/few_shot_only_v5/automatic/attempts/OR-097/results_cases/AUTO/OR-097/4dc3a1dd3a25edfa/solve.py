import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_problem():
    opt_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
    opt_char = pd.read_csv(opt_char_path, sep=',')
    asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
    asset_ref = pd.read_csv(asset_ref_path, sep=',')
    option_ids = list(opt_char['Option'])
    if len(option_ids) != 120:
        raise ValueError(f'Expected 120 options, found {len(option_ids)} in OptionCharacteristics.csv')
    asset_cols = [col for col in asset_ref.columns if re.match('Asset_\\d+', col)]
    asset_ids = [int(col.split('_')[1]) for col in asset_cols]
    if len(asset_ids) != 6:
        raise ValueError(f'Expected 6 assets, found {len(asset_ids)} in Option_AssetReferenceMatrix.csv')
    if 'Option' in asset_ref.columns:
        asset_ref_idx = asset_ref['Option']
    else:
        asset_ref_idx = asset_ref.iloc[:, 0]
    if not all((str(opt) == str(idx) for (opt, idx) in zip(option_ids, asset_ref_idx))):
        asset_ref = asset_ref.set_index(asset_ref_idx)
        asset_ref = asset_ref.loc[option_ids]
    else:
        asset_ref = asset_ref.set_index(asset_ref_idx)
    cost = dict(zip(option_ids, opt_char['Cost']))
    delta = dict(zip(option_ids, opt_char['Delta']))
    gamma = dict(zip(option_ids, opt_char['Gamma']))
    vega = dict(zip(option_ids, opt_char['Vega']))
    maxlong = dict(zip(option_ids, opt_char['MaxLong']))
    maxshort = dict(zip(option_ids, opt_char['MaxShort']))
    A = {}
    for i in option_ids:
        A[i] = {}
        for (j, col) in zip(asset_ids, asset_cols):
            val = asset_ref.loc[i, col]
            if val not in (0, 1):
                raise ValueError(f'Asset reference matrix entry A[{i},{j}] is not 0 or 1: {val}')
            A[i][j] = int(val)
    GREEKS = ['Delta', 'Gamma', 'Vega']
    G_init = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
    G_tol = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
    G_coeff = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
    m = gp.Model('OptionHedging')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb={i: maxshort[i] for i in option_ids}, ub={i: maxlong[i] for i in option_ids}, name='')
    z = m.addVars(option_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
    for i in option_ids:
        m.addConstr(z[i] >= x[i], name=f'z_ge_x_{i}')
        m.addConstr(z[i] >= -x[i], name=f'z_ge_negx_{i}')
    for G in GREEKS:
        expr = G_init[G]
        expr_var = gp.LinExpr()
        for i in option_ids:
            coeff = G_coeff[G][i]
            for j in asset_ids:
                if A[i][j] == 1:
                    expr_var.addTerms(coeff, x[i])
        expr_total = expr + expr_var
        m.addConstr(expr_total <= G_tol[G], name=f'{G}_upper')
        m.addConstr(expr_total >= -G_tol[G], name=f'{G}_lower')
    m.setObjective(gp.quicksum((cost[i] * z[i] for i in option_ids)), gp.GRB.MINIMIZE)
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for i in option_ids:
            print(f'{x[i].VarName} {x[i].X}')
            print(f'{z[i].VarName} {z[i].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_problem()