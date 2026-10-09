import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)

def norm(s):
    return re.sub('\\s+', '', s.strip().casefold())
colmap = {}
for col in df.columns:
    ncol = norm(col)
    if ncol in ['food']:
        colmap['food'] = col
    elif ncol in ['calories']:
        colmap['calories'] = col
    elif ncol in ['protein(g)', 'proteing']:
        colmap['protein'] = col
    elif ncol in ['fat(g)', 'fatg']:
        colmap['fat'] = col
    elif ncol in ['vitaminc(mg)', 'vitamincmg']:
        colmap['vitaminc'] = col
    elif ncol in ['cost']:
        colmap['cost'] = col
required_cols = ['food', 'calories', 'protein', 'fat', 'vitaminc', 'cost']
if not all((k in colmap for k in required_cols)):
    missing = [k for k in required_cols if k not in colmap]
    raise KeyError(f'Missing required columns in CSV: {missing}')
foods = df[colmap['food']].tolist()
if len(set(foods)) != len(foods):
    raise ValueError("Duplicate food identifiers found in 'Food' column.")
calories = pd.to_numeric(df[colmap['calories']], errors='raise')
protein = pd.to_numeric(df[colmap['protein']], errors='raise')
fat = pd.to_numeric(df[colmap['fat']], errors='raise')
vitaminc = pd.to_numeric(df[colmap['vitaminc']], errors='raise')
cost = pd.to_numeric(df[colmap['cost']], errors='raise')
calories_dict = dict(zip(foods, calories))
protein_dict = dict(zip(foods, protein))
fat_dict = dict(zip(foods, fat))
vitaminc_dict = dict(zip(foods, vitaminc))
cost_dict = dict(zip(foods, cost))
m = gp.Model('DietProblem')
x_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost_dict[f] * x_vars[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories_dict[f] * x_vars[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein_dict[f] * x_vars[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitaminc_dict[f] * x_vars[f] for f in foods)) >= 60, name='vitaminc_min')
m.addConstr(gp.quicksum((fat_dict[f] * x_vars[f] for f in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings) ---')
    for f in foods:
        val = x_vars[f].X
        if val > 1e-05:
            print(f'{f}: {val:.4f} servings')
else:
    print(f'No optimal solution found. Status: {m.status}')