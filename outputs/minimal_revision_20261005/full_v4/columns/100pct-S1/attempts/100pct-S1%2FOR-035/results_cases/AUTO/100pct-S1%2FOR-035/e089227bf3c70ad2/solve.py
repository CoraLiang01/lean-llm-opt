import gurobipy as gp
import pandas as pd
import numpy as np
import math
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',')
foods = df['Food'].astype(str).tolist()

def get_param_dict(colname, dtype=float):
    vals = df.set_index(df['Food'].astype(str))[colname]
    if vals.isnull().any():
        raise ValueError(f"Missing values found in column '{colname}' for some foods.")
    return vals.astype(dtype).to_dict()
calories = get_param_dict('Calories', dtype=float)
protein = get_param_dict('Protein(g)', dtype=float)
fat = get_param_dict('Fat(g)', dtype=float)
vitaminc = get_param_dict('VitaminC(mg)', dtype=float)
cost = get_param_dict('Cost', dtype=float)
for food in foods:
    for (pname, pdict) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitaminc), ('Cost', cost)]:
        if food not in pdict:
            raise KeyError(f"Food '{food}' missing parameter '{pname}'.")

def solve_meal_plan(foods, calories, protein, fat, vitaminc, cost):
    m = gp.Model('DietProblem')
    x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[f] * x[f] for f in foods)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((calories[f] * x[f] for f in foods)) >= 2000, name='calories')
    m.addConstr(gp.quicksum((protein[f] * x[f] for f in foods)) >= 50, name='protein')
    m.addConstr(gp.quicksum((vitaminc[f] * x[f] for f in foods)) >= 60, name='vitaminc')
    m.addConstr(gp.quicksum((fat[f] * x[f] for f in foods)) <= 70, name='fat')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_meal_plan(foods, calories, protein, fat, vitaminc, cost)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X:.6f}')
else:
    print(f'Solver status: {m.status}')