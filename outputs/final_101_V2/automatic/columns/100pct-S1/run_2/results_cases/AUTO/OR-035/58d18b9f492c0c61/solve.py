import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S1/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',')
foods = df['Food'].astype(str).tolist()

def get_param_dict(colname, dtype=float):
    if colname not in df.columns:
        raise KeyError(f"Column '{colname}' not found in CSV.")
    vals = df[colname]
    if dtype == float:
        vals = vals.astype(float)
    elif dtype == int:
        vals = vals.astype(int)
    return dict(zip(df['Food'].astype(str), vals))
calories = get_param_dict('Calories', dtype=float)
protein = get_param_dict('Protein(g)', dtype=float)
fat = get_param_dict('Fat(g)', dtype=float)
vitc = get_param_dict('VitaminC(mg)', dtype=float)
cost = get_param_dict('Cost', dtype=float)
for food in foods:
    for param, d in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitc), ('Cost', cost)]:
        if food not in d:
            raise ValueError(f"Missing {param} for food '{food}'.")
m = gp.Model('DietProblem')
x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i] * x[i] for i in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[i] * x[i] for i in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[i] * x[i] for i in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitc[i] * x[i] for i in foods)) >= 60, name='vitc_min')
m.addConstr(gp.quicksum((fat[i] * x[i] for i in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: ${m.objVal:.2f}')
    print('Meal plan (servings per food):')
    for i in foods:
        xi = x[i].X
        if xi > 1e-05:
            print(f'  {i}: {xi:.3f} servings')
else:
    print(f'No optimal solution found. Status: {m.status}')