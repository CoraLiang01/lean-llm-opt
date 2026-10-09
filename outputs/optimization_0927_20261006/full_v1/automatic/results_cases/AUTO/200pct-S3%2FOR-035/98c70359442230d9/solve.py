import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
foods = df['Food'].tolist()

def get_numeric_col(df, colname, foods):
    col_candidates = [c for c in df.columns if c.strip().casefold() == colname.strip().casefold()]
    if not col_candidates:
        raise KeyError(f"Column '{colname}' not found in CSV columns: {df.columns.tolist()}")
    col = col_candidates[0]
    vals = df[col].astype(float)
    return dict(zip(df['Food'], vals))
calories = get_numeric_col(df, 'Calories', foods)
protein = get_numeric_col(df, 'Protein(g)', foods)
fat = get_numeric_col(df, 'Fat(g)', foods)
vitamin_c = get_numeric_col(df, 'VitaminC(mg)', foods)
cost = get_numeric_col(df, 'Cost', foods)
for f in foods:
    for (pname, p) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
        if f not in p:
            raise ValueError(f"Missing {pname} data for food '{f}'.")
m = gp.Model('MealPlanMinCost')
x_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * x_vars[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * x_vars[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[f] * x_vars[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[f] * x_vars[f] for f in foods)) >= 60, name='vitaminC_min')
m.addConstr(gp.quicksum((fat[f] * x_vars[f] for f in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings) ---')
    for f in foods:
        val = x_vars[f].X
        if val > 1e-05:
            print(f'  {f}: {val:.4f} servings')
    total_cal = sum((calories[f] * x_vars[f].X for f in foods))
    total_prot = sum((protein[f] * x_vars[f].X for f in foods))
    total_fat = sum((fat[f] * x_vars[f].X for f in foods))
    total_vitc = sum((vitamin_c[f] * x_vars[f].X for f in foods))
    print('\n--- Achieved Nutrition Totals ---')
    print(f'  Calories:   {total_cal:.2f} kcal')
    print(f'  Protein:    {total_prot:.2f} g')
    print(f'  Fat:        {total_fat:.2f} g')
    print(f'  Vitamin C:  {total_vitc:.2f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')