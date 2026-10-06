import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S2/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',')
foods = df['Food'].astype(str).tolist()

def build_param_dict(colname, dtype=float):
    col_actual = None
    for c in df.columns:
        if c.strip().casefold() == colname.strip().casefold():
            col_actual = c
            break
    if col_actual is None:
        raise KeyError(f"Required column '{colname}' not found in CSV.")
    vals = df[col_actual].values
    if dtype == float:
        vals = vals.astype(float)
    elif dtype == int:
        vals = vals.astype(int)
    return dict(zip(foods, vals))
calories = build_param_dict('Calories', dtype=float)
protein = build_param_dict('Protein(g)', dtype=float)
fat = build_param_dict('Fat(g)', dtype=float)
vitamin_c = build_param_dict('VitaminC(mg)', dtype=float)
cost = build_param_dict('Cost', dtype=float)
for food in foods:
    for param, d in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
        if food not in d:
            raise ValueError(f"Food '{food}' missing parameter '{param}'.")
m = gp.Model('DietProblem')
x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i] * x[i] for i in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[i] * x[i] for i in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[i] * x[i] for i in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[i] * x[i] for i in foods)) >= 60, name='vitaminc_min')
m.addConstr(gp.quicksum((fat[i] * x[i] for i in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: ${m.objVal:.2f}')
    print('Meal plan (servings per food):')
    for i in foods:
        if x[i].X > 1e-05:
            print(f'  {i}: {x[i].X:.4f} servings')
    total_cal = sum((calories[i] * x[i].X for i in foods))
    total_prot = sum((protein[i] * x[i].X for i in foods))
    total_fat = sum((fat[i] * x[i].X for i in foods))
    total_vitc = sum((vitamin_c[i] * x[i].X for i in foods))
    print('\nAchieved totals:')
    print(f'  Calories: {total_cal:.2f} kcal')
    print(f'  Protein: {total_prot:.2f} g')
    print(f'  Fat: {total_fat:.2f} g')
    print(f'  Vitamin C: {total_vitc:.2f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')