import gurobipy as gp
import pandas as pd
import numpy as np
import math
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',')
foods = df['Food'].astype(str).tolist()

def get_param_dict(colname, dtype=float):
    if colname not in df.columns:
        raise KeyError(f'Missing required column: {colname}')
    vals = df[colname].values
    if len(vals) != len(foods):
        raise ValueError(f'Length mismatch for {colname}: {len(vals)} vs {len(foods)}')
    return dict(zip(foods, vals.astype(dtype)))
calories = get_param_dict('Calories', dtype=float)
protein = get_param_dict('Protein(g)', dtype=float)
fat = get_param_dict('Fat(g)', dtype=float)
vitamin_c = get_param_dict('VitaminC(mg)', dtype=float)
cost = get_param_dict('Cost', dtype=float)
for (param, d) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
    if set(d.keys()) != set(foods):
        raise ValueError(f'Parameter {param} missing foods: {set(foods) - set(d.keys())}')

def solve_meal_plan(foods, calories, protein, fat, vitamin_c, cost):
    m = gp.Model('DietProblem')
    m.Params.MIPGap = 0.0001
    x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i] * x[i] for i in foods)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((calories[i] * x[i] for i in foods)) >= 2000, name='calories')
    m.addConstr(gp.quicksum((protein[i] * x[i] for i in foods)) >= 50, name='protein')
    m.addConstr(gp.quicksum((vitamin_c[i] * x[i] for i in foods)) >= 60, name='vitamin_c')
    m.addConstr(gp.quicksum((fat[i] * x[i] for i in foods)) <= 70, name='fat')
    m.optimize()
    return m
m = solve_meal_plan(foods, calories, protein, fat, vitamin_c, cost)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X:.6f}')
else:
    print(f'Solver status: {m.status}')