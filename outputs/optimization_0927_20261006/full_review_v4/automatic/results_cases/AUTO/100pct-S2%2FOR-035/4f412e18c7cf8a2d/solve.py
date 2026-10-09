import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
foods = df['Food'].tolist()

def to_float_series(df, col):
    col = col.strip()
    try:
        return df[col].astype(float)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{col}' to float: {e}")
calories = to_float_series(df, 'Calories')
protein = to_float_series(df, 'Protein(g)')
fat = to_float_series(df, 'Fat(g)')
vitamin_c = to_float_series(df, 'VitaminC(mg)')
cost = to_float_series(df, 'Cost')
calories_dict = dict(zip(foods, calories))
protein_dict = dict(zip(foods, protein))
fat_dict = dict(zip(foods, fat))
vitamin_c_dict = dict(zip(foods, vitamin_c))
cost_dict = dict(zip(foods, cost))
for food in foods:
    for (d, name) in [(calories_dict, 'Calories'), (protein_dict, 'Protein(g)'), (fat_dict, 'Fat(g)'), (vitamin_c_dict, 'VitaminC(mg)'), (cost_dict, 'Cost')]:
        if food not in d:
            raise KeyError(f"Missing {name} data for food '{food}'.")
m = gp.Model('OneDayMealPlan')
servings_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost_dict[food] * servings_vars[food] for food in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories_dict[food] * servings_vars[food] for food in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein_dict[food] * servings_vars[food] for food in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c_dict[food] * servings_vars[food] for food in foods)) >= 60, name='vitamin_c_min')
m.addConstr(gp.quicksum((fat_dict[food] * servings_vars[food] for food in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for food in foods:
        val = servings_vars[food].X
        if val > 1e-05:
            print(f'{food}: {val:.4f} servings')
    total_cal = sum((calories_dict[food] * servings_vars[food].X for food in foods))
    total_prot = sum((protein_dict[food] * servings_vars[food].X for food in foods))
    total_fat = sum((fat_dict[food] * servings_vars[food].X for food in foods))
    total_vitc = sum((vitamin_c_dict[food] * servings_vars[food].X for food in foods))
    print('\n--- Achieved Nutrition Totals ---')
    print(f'Calories: {total_cal:.2f} kcal')
    print(f'Protein: {total_prot:.2f} g')
    print(f'Fat: {total_fat:.2f} g')
    print(f'Vitamin C: {total_vitc:.2f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')