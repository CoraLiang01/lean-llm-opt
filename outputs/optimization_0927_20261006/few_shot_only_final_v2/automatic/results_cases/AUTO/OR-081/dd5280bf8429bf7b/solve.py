import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv'
cost_df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
foods = cost_df['Food'].tolist()

def to_float_col(df, col):
    try:
        return pd.Series(df[col], index=df['Food']).astype(float).to_dict()
    except Exception as e:
        raise ValueError(f"Column '{col}' could not be converted to float: {e}")
calories = to_float_col(cost_df, 'Calories')
protein = to_float_col(cost_df, 'Protein(g)')
fat = to_float_col(cost_df, 'Fat(g)')
vitaminc = to_float_col(cost_df, 'VitaminC(mg)')
cost = to_float_col(cost_df, 'Cost')
for f in foods:
    for (param, d) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitaminc), ('Cost', cost)]:
        if f not in d or not np.isfinite(d[f]):
            raise ValueError(f"Missing or invalid value for '{param}' in food '{f}'.")
m = gp.Model('MealPlan')
x_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * x_vars[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * x_vars[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[f] * x_vars[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitaminc[f] * x_vars[f] for f in foods)) >= 60, name='vitaminc_min')
m.addConstr(gp.quicksum((fat[f] * x_vars[f] for f in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: ${m.objVal:.2f}')
    print('--- Optimal Meal Plan (servings per food) ---')
    for f in foods:
        servings = x_vars[f].X
        if servings > 1e-05:
            print(f'  {f}: {servings:.4f} servings')
    total_cal = sum((calories[f] * x_vars[f].X for f in foods))
    total_prot = sum((protein[f] * x_vars[f].X for f in foods))
    total_fat = sum((fat[f] * x_vars[f].X for f in foods))
    total_vitc = sum((vitaminc[f] * x_vars[f].X for f in foods))
    print('\n--- Nutrient Totals ---')
    print(f'  Calories: {total_cal:.2f} kcal')
    print(f'  Protein: {total_prot:.2f} g')
    print(f'  Fat: {total_fat:.2f} g')
    print(f'  Vitamin C: {total_vitc:.2f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')