import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv', sep=',')

def normalize_col(col):
    return col.strip().casefold().replace(' ', '').replace('(', '').replace(')', '')
col_map = {normalize_col(c): c for c in cost_df.columns}
required_cols = {'food': ['food'], 'calories': ['calories'], 'protein': ['protein(g)'], 'fat': ['fat(g)'], 'vitaminc': ['vitaminc(mg)'], 'cost': ['cost']}
actual_cols = {}
for (key, options) in required_cols.items():
    found = False
    for opt in options:
        norm_opt = normalize_col(opt)
        for (norm_col, orig_col) in col_map.items():
            if norm_col == norm_opt:
                actual_cols[key] = orig_col
                found = True
                break
        if found:
            break
    if not found:
        raise KeyError(f"Required column '{key}' not found in cost.csv.")
foods = cost_df[actual_cols['food']].astype(str).tolist()
n_foods = len(foods)

def build_param_dict(colname):
    s = cost_df.set_index(actual_cols['food'])[colname]
    if s.isnull().any():
        raise ValueError(f"Missing values in column '{colname}'.")
    return s.astype(float).to_dict()
calories = build_param_dict(actual_cols['calories'])
protein = build_param_dict(actual_cols['protein'])
fat = build_param_dict(actual_cols['fat'])
vitaminc = build_param_dict(actual_cols['vitaminc'])
cost = build_param_dict(actual_cols['cost'])
for f in foods:
    for (param, d) in [('calories', calories), ('protein', protein), ('fat', fat), ('vitaminc', vitaminc), ('cost', cost)]:
        if f not in d:
            raise KeyError(f"Food '{f}' missing parameter '{param}'.")

def solve_nutrition():
    m = gp.Model('OneDayMealPlan')
    m.Params.MIPGap = 0.0001
    x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[f] * x[f] for f in foods)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((calories[f] * x[f] for f in foods)) >= 2000, name='cal_min')
    m.addConstr(gp.quicksum((protein[f] * x[f] for f in foods)) >= 50, name='prot_min')
    m.addConstr(gp.quicksum((vitaminc[f] * x[f] for f in foods)) >= 60, name='vitc_min')
    m.addConstr(gp.quicksum((fat[f] * x[f] for f in foods)) <= 70, name='fat_max')
    m.optimize()
    return m
m = solve_nutrition()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X:.6f}')
else:
    print(f'Solver status: {m.status}')