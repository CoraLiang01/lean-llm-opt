import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)

def norm_col(df, pattern):
    for col in df.columns:
        if re.fullmatch(pattern, col.strip().replace(' ', '').replace('_', '').replace('-', '').casefold()):
            return col
    for col in df.columns:
        if pattern in col.strip().replace(' ', '').replace('_', '').replace('-', '').casefold():
            return col
    raise KeyError(f"Could not find a column matching pattern '{pattern}'")
food_col = 'Food'
calories_col = norm_col(df, 'calories')
protein_col = norm_col(df, 'proteing')
fat_col = norm_col(df, 'fatg')
vitc_col = norm_col(df, 'vitamincmg')
cost_col = norm_col(df, 'cost')
foods = df[food_col].astype(str).tolist()

def float_series(series, name):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Column '{name}' could not be converted to float: {e}")
calories = dict(zip(foods, float_series(df[calories_col], calories_col)))
protein = dict(zip(foods, float_series(df[protein_col], protein_col)))
fat = dict(zip(foods, float_series(df[fat_col], fat_col)))
vitc = dict(zip(foods, float_series(df[vitc_col], vitc_col)))
cost = dict(zip(foods, float_series(df[cost_col], cost_col)))
for f in foods:
    for (dct, nm) in [(calories, 'Calories'), (protein, 'Protein(g)'), (fat, 'Fat(g)'), (vitc, 'VitaminC(mg)'), (cost, 'Cost')]:
        if f not in dct:
            raise KeyError(f"Missing {nm} data for food '{f}'")
m = gp.Model('MealPlanMinCost')
x_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * x_vars[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * x_vars[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[f] * x_vars[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitc[f] * x_vars[f] for f in foods)) >= 60, name='vitc_min')
m.addConstr(gp.quicksum((fat[f] * x_vars[f] for f in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for f in foods:
        val = x_vars[f].X
        if val > 1e-05:
            print(f'  {f}: {val:.4f} servings')
    total_cal = sum((calories[f] * x_vars[f].X for f in foods))
    total_prot = sum((protein[f] * x_vars[f].X for f in foods))
    total_vitc = sum((vitc[f] * x_vars[f].X for f in foods))
    total_fat = sum((fat[f] * x_vars[f].X for f in foods))
    print('\n--- Achieved Nutrient Totals ---')
    print(f'  Calories:   {total_cal:.2f} kcal')
    print(f'  Protein:    {total_prot:.2f} g')
    print(f'  Vitamin C:  {total_vitc:.2f} mg')
    print(f'  Fat:        {total_fat:.2f} g')
else:
    print(f'No optimal solution found. Status: {m.status}')