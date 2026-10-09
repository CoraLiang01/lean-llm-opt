import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
food_ids = df['Food'].tolist()

def get_numeric_col(colname):
    col_candidates = [c for c in df.columns if c.strip().casefold() == colname.strip().casefold()]
    if not col_candidates:
        raise KeyError(f"Required column '{colname}' not found in CSV columns: {list(df.columns)}")
    col = col_candidates[0]
    vals = pd.to_numeric(df[col], errors='raise')
    if len(vals) != len(food_ids):
        raise ValueError(f"Length mismatch for column '{colname}' and food_ids")
    return dict(zip(food_ids, vals))
cost = get_numeric_col('Cost')
calories = get_numeric_col('Calories')
protein = get_numeric_col('Protein(g)')
fat = get_numeric_col('Fat(g)')
vitamin_c = get_numeric_col('VitaminC(mg)')
m = gp.Model('MealPlanMinCost')
x_vars = m.addVars(food_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i] * x_vars[i] for i in food_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[i] * x_vars[i] for i in food_ids)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[i] * x_vars[i] for i in food_ids)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[i] * x_vars[i] for i in food_ids)) >= 60, name='vitaminC_min')
m.addConstr(gp.quicksum((fat[i] * x_vars[i] for i in food_ids)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for i in food_ids:
        val = x_vars[i].X
        if val > 1e-05:
            print(f'{i}: {val:.4f} servings')
    total_cal = sum((calories[i] * x_vars[i].X for i in food_ids))
    total_prot = sum((protein[i] * x_vars[i].X for i in food_ids))
    total_fat = sum((fat[i] * x_vars[i].X for i in food_ids))
    total_vitc = sum((vitamin_c[i] * x_vars[i].X for i in food_ids))
    print('\n--- Achieved Totals ---')
    print(f'Calories: {total_cal:.2f} kcal')
    print(f'Protein: {total_prot:.2f} g')
    print(f'Fat: {total_fat:.2f} g')
    print(f'Vitamin C: {total_vitc:.2f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')