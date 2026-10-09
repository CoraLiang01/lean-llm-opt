import gurobipy as gp
import pandas as pd
import numpy as np
import re

def find_col(df, pattern):
    pat = re.compile(pattern, re.IGNORECASE)
    for col in df.columns:
        if pat.fullmatch(col.strip()):
            return col
    for col in df.columns:
        if pat.search(col.strip()):
            return col
    raise KeyError(f"Could not find a column matching pattern '{pattern}'")
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
food_col = find_col(df, 'Food')
calories_col = find_col(df, 'Calories')
protein_col = find_col(df, 'Protein\\(g\\)')
fat_col = find_col(df, 'Fat\\(g\\)')
vitc_col = find_col(df, 'VitaminC\\(mg\\)')
cost_col = find_col(df, 'Cost')
foods = df[food_col].astype(str).tolist()
calories = pd.to_numeric(df[calories_col], errors='raise')
protein = pd.to_numeric(df[protein_col], errors='raise')
fat = pd.to_numeric(df[fat_col], errors='raise')
vitc = pd.to_numeric(df[vitc_col], errors='raise')
cost = pd.to_numeric(df[cost_col], errors='raise')
calories_dict = dict(zip(foods, calories))
protein_dict = dict(zip(foods, protein))
fat_dict = dict(zip(foods, fat))
vitc_dict = dict(zip(foods, vitc))
cost_dict = dict(zip(foods, cost))
m = gp.Model('DietProblem')
x_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost_dict[i] * x_vars[i] for i in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories_dict[i] * x_vars[i] for i in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein_dict[i] * x_vars[i] for i in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitc_dict[i] * x_vars[i] for i in foods)) >= 60, name='vitc_min')
m.addConstr(gp.quicksum((fat_dict[i] * x_vars[i] for i in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings) ---')
    for i in foods:
        val = x_vars[i].X
        if val > 1e-05:
            print(f'{i}: {val:.3f} servings')
    total_cal = sum((calories_dict[i] * x_vars[i].X for i in foods))
    total_prot = sum((protein_dict[i] * x_vars[i].X for i in foods))
    total_fat = sum((fat_dict[i] * x_vars[i].X for i in foods))
    total_vitc = sum((vitc_dict[i] * x_vars[i].X for i in foods))
    print('\n--- Achieved Nutrient Totals ---')
    print(f'Calories: {total_cal:.1f} kcal')
    print(f'Protein: {total_prot:.1f} g')
    print(f'Fat: {total_fat:.1f} g')
    print(f'Vitamin C: {total_vitc:.1f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')