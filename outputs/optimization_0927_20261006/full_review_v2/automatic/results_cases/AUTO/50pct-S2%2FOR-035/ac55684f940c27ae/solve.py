import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
foods = df['Food'].astype(str).tolist()

def to_float_series(series, colname):
    try:
        return pd.to_numeric(series, errors='raise')
    except Exception as e:
        raise ValueError(f"Column '{colname}' contains non-numeric values or missing data.") from e
calories = to_float_series(df['Calories'], 'Calories')
protein = to_float_series(df['Protein(g)'], 'Protein(g)')
fat = to_float_series(df['Fat(g)'], 'Fat(g)')
vitamin_c = to_float_series(df['VitaminC(mg)'], 'VitaminC(mg)')
cost = to_float_series(df['Cost'], 'Cost')
calories_dict = dict(zip(foods, calories))
protein_dict = dict(zip(foods, protein))
fat_dict = dict(zip(foods, fat))
vitamin_c_dict = dict(zip(foods, vitamin_c))
cost_dict = dict(zip(foods, cost))
for food in foods:
    for (dct, name) in [(calories_dict, 'Calories'), (protein_dict, 'Protein(g)'), (fat_dict, 'Fat(g)'), (vitamin_c_dict, 'VitaminC(mg)'), (cost_dict, 'Cost')]:
        if food not in dct or pd.isnull(dct[food]):
            raise ValueError(f"Missing or invalid {name} value for food '{food}'.")
m = gp.Model('DietProblem')
x_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost_dict[food] * x_vars[food] for food in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories_dict[food] * x_vars[food] for food in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein_dict[food] * x_vars[food] for food in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c_dict[food] * x_vars[food] for food in foods)) >= 60, name='vitaminc_min')
m.addConstr(gp.quicksum((fat_dict[food] * x_vars[food] for food in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for food in foods:
        servings = x_vars[food].X
        if servings > 1e-05:
            print(f'{food}: {servings:.3f} servings')
    total_calories = sum((calories_dict[food] * x_vars[food].X for food in foods))
    total_protein = sum((protein_dict[food] * x_vars[food].X for food in foods))
    total_vitamin_c = sum((vitamin_c_dict[food] * x_vars[food].X for food in foods))
    total_fat = sum((fat_dict[food] * x_vars[food].X for food in foods))
    print('\n--- Achieved Nutrient Totals ---')
    print(f'Calories: {total_calories:.1f} kcal')
    print(f'Protein: {total_protein:.1f} g')
    print(f'Vitamin C: {total_vitamin_c:.1f} mg')
    print(f'Fat: {total_fat:.1f} g')
else:
    print(f'No optimal solution found. Status: {m.status}')