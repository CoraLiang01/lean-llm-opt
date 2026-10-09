import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Others_example/Others4/cost.csv'
cost_df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
foods = cost_df['Food'].tolist()

def get_numeric_col(df, colname, foods, float_type=float):
    col_candidates = [c for c in df.columns if c.strip().casefold() == colname.strip().casefold()]
    if not col_candidates:
        raise KeyError(f"Required column '{colname}' not found in CSV.")
    col = col_candidates[0]
    vals = {}
    for (idx, row) in df.iterrows():
        food = row['Food']
        val_str = row[col]
        try:
            val = float_type(val_str)
        except Exception as e:
            raise ValueError(f"Could not convert value '{val_str}' in column '{col}' for food '{food}': {e}")
        vals[food] = val
    missing = set(foods) - set(vals.keys())
    if missing:
        raise ValueError(f"Missing values for foods: {missing} in column '{col}'")
    return vals
calories = get_numeric_col(cost_df, 'Calories', foods, float_type=float)
protein = get_numeric_col(cost_df, 'Protein(g)', foods, float_type=float)
fat = get_numeric_col(cost_df, 'Fat(g)', foods, float_type=float)
vitamin_c = get_numeric_col(cost_df, 'VitaminC(mg)', foods, float_type=float)
cost = get_numeric_col(cost_df, 'Cost', foods, float_type=float)
m = gp.Model('DietOptimization')
x_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * x_vars[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * x_vars[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[f] * x_vars[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[f] * x_vars[f] for f in foods)) >= 60, name='vitaminC_min')
m.addConstr(gp.quicksum((fat[f] * x_vars[f] for f in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for f in foods:
        servings = x_vars[f].X
        if servings > 1e-05:
            print(f'{f}: {servings:.4f} servings')
    total_cal = sum((calories[f] * x_vars[f].X for f in foods))
    total_prot = sum((protein[f] * x_vars[f].X for f in foods))
    total_fat = sum((fat[f] * x_vars[f].X for f in foods))
    total_vitc = sum((vitamin_c[f] * x_vars[f].X for f in foods))
    print('\n--- Nutrient Totals ---')
    print(f'Calories: {total_cal:.2f} kcal')
    print(f'Protein: {total_prot:.2f} g')
    print(f'Fat: {total_fat:.2f} g')
    print(f'Vitamin C: {total_vitc:.2f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')