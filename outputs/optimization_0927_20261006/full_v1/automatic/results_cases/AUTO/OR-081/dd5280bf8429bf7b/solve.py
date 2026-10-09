import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv', sep=',', dtype=str, keep_default_na=False)
foods = cost_df['Food'].tolist()

def to_float_col(df, col):
    try:
        return df[col].astype(float)
    except Exception as e:
        raise ValueError(f"Column '{col}' could not be converted to float: {e}")
calories = dict(zip(foods, to_float_col(cost_df, 'Calories')))
protein = dict(zip(foods, to_float_col(cost_df, 'Protein(g)')))
fat = dict(zip(foods, to_float_col(cost_df, 'Fat(g)')))
vitamin_c = dict(zip(foods, to_float_col(cost_df, 'VitaminC(mg)')))
cost = dict(zip(foods, to_float_col(cost_df, 'Cost')))
for f in foods:
    for (param, d) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
        if f not in d:
            raise KeyError(f"Missing {param} data for food '{f}'.")
m = gp.Model('MealPlanMinCost')
servings_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * servings_vars[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * servings_vars[f] for f in foods)) >= 2000, name='calories')
m.addConstr(gp.quicksum((protein[f] * servings_vars[f] for f in foods)) >= 50, name='protein')
m.addConstr(gp.quicksum((vitamin_c[f] * servings_vars[f] for f in foods)) >= 60, name='vitaminC')
m.addConstr(gp.quicksum((fat[f] * servings_vars[f] for f in foods)) <= 70, name='fat')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: ${m.objVal:.2f}')
    print('--- Optimal Meal Plan (servings per food) ---')
    for f in foods:
        val = servings_vars[f].X
        if val > 1e-05:
            print(f'{f}: {val:.4f} servings')
    total_cal = sum((calories[f] * servings_vars[f].X for f in foods))
    total_prot = sum((protein[f] * servings_vars[f].X for f in foods))
    total_fat = sum((fat[f] * servings_vars[f].X for f in foods))
    total_vitc = sum((vitamin_c[f] * servings_vars[f].X for f in foods))
    print('\n--- Achieved Nutrition Totals ---')
    print(f'Calories: {total_cal:.2f} kcal')
    print(f'Protein: {total_prot:.2f} g')
    print(f'Fat: {total_fat:.2f} g')
    print(f'Vitamin C: {total_vitc:.2f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')