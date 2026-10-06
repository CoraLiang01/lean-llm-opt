import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',')
expected_columns = ['Food', 'Calories', 'Protein(g)', 'Fat(g)', 'VitaminC(mg)', 'Cost']
if not all((col in df.columns for col in expected_columns)):
    raise ValueError(f'Missing columns in cost.csv. Expected columns: {expected_columns}')
foods = df['Food'].astype(str).tolist()
if len(set(foods)) != len(foods):
    raise ValueError('Duplicate Food identifiers found in cost.csv.')
calories = dict(zip(foods, df['Calories'].astype(float)))
protein = dict(zip(foods, df['Protein(g)'].astype(float)))
fat = dict(zip(foods, df['Fat(g)'].astype(float)))
vitaminc = dict(zip(foods, df['VitaminC(mg)'].astype(float)))
cost = dict(zip(foods, df['Cost'].astype(float)))
for f in foods:
    for (d, name) in [(calories, 'Calories'), (protein, 'Protein(g)'), (fat, 'Fat(g)'), (vitaminc, 'VitaminC(mg)'), (cost, 'Cost')]:
        if f not in d or not np.isfinite(d[f]):
            raise ValueError(f"Missing or invalid {name} for food '{f}'.")

def solve_nutrition_lp():
    m = gp.Model('MealPlanMinCost')
    m.Params.MIPGap = 0.0001
    x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[f] * x[f] for f in foods)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((calories[f] * x[f] for f in foods)) >= 2000, name='cal_min')
    m.addConstr(gp.quicksum((protein[f] * x[f] for f in foods)) >= 50, name='prot_min')
    m.addConstr(gp.quicksum((vitaminc[f] * x[f] for f in foods)) >= 60, name='vitc_min')
    m.addConstr(gp.quicksum((fat[f] * x[f] for f in foods)) <= 70, name='fat_max')
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal:.6f}')
        for f in foods:
            var = x[f]
            print(f'{var.VarName} {var.X:.6f}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_nutrition_lp()