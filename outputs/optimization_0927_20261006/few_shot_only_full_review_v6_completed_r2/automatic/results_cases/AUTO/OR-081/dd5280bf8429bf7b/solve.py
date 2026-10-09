import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv'
cost_df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['Food', 'Calories', 'Protein(g)', 'Fat(g)', 'VitaminC(mg)', 'Cost']
for col in required_columns:
    if col not in cost_df.columns:
        raise KeyError(f"Required column '{col}' not found in cost.csv.")
foods = cost_df['Food'].tolist()

def to_float_series(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Column '{colname}' contains non-numeric values.") from e
calories = dict(zip(foods, to_float_series(cost_df['Calories'], 'Calories')))
protein = dict(zip(foods, to_float_series(cost_df['Protein(g)'], 'Protein(g)')))
fat = dict(zip(foods, to_float_series(cost_df['Fat(g)'], 'Fat(g)')))
vitamin_c = dict(zip(foods, to_float_series(cost_df['VitaminC(mg)'], 'VitaminC(mg)')))
cost = dict(zip(foods, to_float_series(cost_df['Cost'], 'Cost')))
m = gp.Model('MealPlanMinCost')
x_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i] * x_vars[i] for i in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[i] * x_vars[i] for i in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[i] * x_vars[i] for i in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[i] * x_vars[i] for i in foods)) >= 60, name='vitaminC_min')
m.addConstr(gp.quicksum((fat[i] * x_vars[i] for i in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: ${m.objVal:.2f}')
    print('--- Optimal Meal Plan (servings per food) ---')
    for i in foods:
        val = x_vars[i].X
        if val > 1e-05:
            print(f'{i}: {val:.3f} servings')
    total_cal = sum((calories[i] * x_vars[i].X for i in foods))
    total_prot = sum((protein[i] * x_vars[i].X for i in foods))
    total_fat = sum((fat[i] * x_vars[i].X for i in foods))
    total_vitc = sum((vitamin_c[i] * x_vars[i].X for i in foods))
    print('\n--- Achieved Nutrition Totals ---')
    print(f'Calories: {total_cal:.1f} kcal')
    print(f'Protein: {total_prot:.1f} g')
    print(f'Fat: {total_fat:.1f} g')
    print(f'Vitamin C: {total_vitc:.1f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')