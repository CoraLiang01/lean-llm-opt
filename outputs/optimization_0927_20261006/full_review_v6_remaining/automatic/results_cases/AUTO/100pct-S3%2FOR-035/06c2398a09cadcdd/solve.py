import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
required_columns = {'Food': None, 'Calories': None, 'Protein(g)': None, 'Fat(g)': None, 'VitaminC(mg)': None, 'Cost': None}

def find_col(df, pattern):
    for col in df.columns:
        if re.fullmatch(pattern, col.strip(), re.IGNORECASE):
            return col
    for col in df.columns:
        if pattern.strip().casefold() == col.strip().casefold():
            return col
    raise KeyError(f"Could not find a column matching pattern '{pattern}'")
for key in required_columns:
    required_columns[key] = find_col(df, key)
food_col = required_columns['Food']
df['Food_id'] = df[food_col].astype(str)
df = df.set_index('Food_id', drop=False)
foods = list(df.index)
calories = df[required_columns['Calories']].astype(float).to_dict()
protein = df[required_columns['Protein(g)']].astype(float).to_dict()
fat = df[required_columns['Fat(g)']].astype(float).to_dict()
vitaminc = df[required_columns['VitaminC(mg)']].astype(float).to_dict()
cost = df[required_columns['Cost']].astype(float).to_dict()
for f in foods:
    for (param, d) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitaminc), ('Cost', cost)]:
        if f not in d or pd.isnull(d[f]):
            raise ValueError(f"Missing or invalid value for {param} in food '{f}'.")
m = gp.Model('MealPlanMinCost')
x_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * x_vars[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * x_vars[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[f] * x_vars[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitaminc[f] * x_vars[f] for f in foods)) >= 60, name='vitaminc_min')
m.addConstr(gp.quicksum((fat[f] * x_vars[f] for f in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for f in foods:
        val = x_vars[f].X
        if val > 1e-05:
            print(f'{f}: {val:.4f} servings')
    print('\nNutrient totals:')
    total_cal = sum((calories[f] * x_vars[f].X for f in foods))
    total_prot = sum((protein[f] * x_vars[f].X for f in foods))
    total_fat = sum((fat[f] * x_vars[f].X for f in foods))
    total_vitc = sum((vitaminc[f] * x_vars[f].X for f in foods))
    print(f'  Calories:   {total_cal:.2f} kcal')
    print(f'  Protein:    {total_prot:.2f} g')
    print(f'  Fat:        {total_fat:.2f} g')
    print(f'  Vitamin C:  {total_vitc:.2f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')