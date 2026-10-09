import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Others_example/Others4/cost.csv'
cost_df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['Food', 'Calories', 'Protein(g)', 'Fat(g)', 'VitaminC(mg)', 'Cost']
for col in required_columns:
    if col not in cost_df.columns:
        raise KeyError(f"Required column '{col}' not found in cost.csv")
foods = cost_df['Food'].tolist()

def to_float_series(df, col):
    try:
        return df[col].astype(float)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{col}' to float: {e}")
calories = dict(zip(foods, to_float_series(cost_df, 'Calories')))
protein = dict(zip(foods, to_float_series(cost_df, 'Protein(g)')))
fat = dict(zip(foods, to_float_series(cost_df, 'Fat(g)')))
vitamin_c = dict(zip(foods, to_float_series(cost_df, 'VitaminC(mg)')))
cost = dict(zip(foods, to_float_series(cost_df, 'Cost')))
for food in foods:
    for (param, d) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
        if food not in d:
            raise KeyError(f"Missing {param} for food '{food}'")
m = gp.Model('DietOptimization')
servings_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[food] * servings_vars[food] for food in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[food] * servings_vars[food] for food in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[food] * servings_vars[food] for food in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[food] * servings_vars[food] for food in foods)) >= 60, name='vitamin_c_min')
m.addConstr(gp.quicksum((fat[food] * servings_vars[food] for food in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for food in foods:
        val = servings_vars[food].X
        if val > 1e-05:
            print(f'{food}: {val:.3f} servings')
    total_cal = sum((calories[food] * servings_vars[food].X for food in foods))
    total_prot = sum((protein[food] * servings_vars[food].X for food in foods))
    total_fat = sum((fat[food] * servings_vars[food].X for food in foods))
    total_vitc = sum((vitamin_c[food] * servings_vars[food].X for food in foods))
    print('\n--- Achieved Nutrient Totals ---')
    print(f'Calories: {total_cal:.1f} kcal')
    print(f'Protein: {total_prot:.1f} g')
    print(f'Fat: {total_fat:.1f} g')
    print(f'Vitamin C: {total_vitc:.1f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')