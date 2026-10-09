import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',')
foods = df['Food'].astype(str).tolist()

def get_param_dict(colname, dtype=float):
    if colname not in df.columns:
        raise KeyError(f"Column '{colname}' not found in CSV.")
    vals = df[colname].values
    if len(vals) != len(foods):
        raise ValueError(f"Length mismatch for column '{colname}' and foods.")
    return {str(food): dtype(val) for (food, val) in zip(foods, vals)}
calories = get_param_dict('Calories', float)
protein = get_param_dict('Protein(g)', float)
fat = get_param_dict('Fat(g)', float)
vitamin_c = get_param_dict('VitaminC(mg)', float)
cost = get_param_dict('Cost', float)
m = gp.Model('DietProblem')
x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i] * x[i] for i in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[i] * x[i] for i in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[i] * x[i] for i in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[i] * x[i] for i in foods)) >= 60, name='vitaminC_min')
m.addConstr(gp.quicksum((fat[i] * x[i] for i in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for i in foods:
        xi = x[i].X
        if xi > 1e-05:
            print(f'{i}: {xi:.4f} servings')
    total_cal = sum((calories[i] * x[i].X for i in foods))
    total_prot = sum((protein[i] * x[i].X for i in foods))
    total_fat = sum((fat[i] * x[i].X for i in foods))
    total_vitc = sum((vitamin_c[i] * x[i].X for i in foods))
    print('\n--- Achieved Nutrition Totals ---')
    print(f'Calories:   {total_cal:.2f} kcal')
    print(f'Protein:    {total_prot:.2f} g')
    print(f'Fat:        {total_fat:.2f} g')
    print(f'Vitamin C:  {total_vitc:.2f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')