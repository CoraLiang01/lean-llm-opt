import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
foods = df['Food'].tolist()

def to_float_series(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{colname}' to float: {e}")
calories = dict(zip(foods, to_float_series(df['Calories'], 'Calories')))
protein = dict(zip(foods, to_float_series(df['Protein(g)'], 'Protein(g)')))
fat = dict(zip(foods, to_float_series(df['Fat(g)'], 'Fat(g)')))
vitamin_c = dict(zip(foods, to_float_series(df['VitaminC(mg)'], 'VitaminC(mg)')))
cost = dict(zip(foods, to_float_series(df['Cost'], 'Cost')))
for food in foods:
    for (param, d) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
        if food not in d:
            raise KeyError(f"Missing {param} for food '{food}'")
        if not np.isfinite(d[food]):
            raise ValueError(f"Non-finite {param} for food '{food}'")
m = gp.Model('MealPlan')
servings_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * servings_vars[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * servings_vars[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[f] * servings_vars[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[f] * servings_vars[f] for f in foods)) >= 60, name='vitaminC_min')
m.addConstr(gp.quicksum((fat[f] * servings_vars[f] for f in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for f in foods:
        x = servings_vars[f].X
        if x > 1e-05:
            print(f'{f}: {x:.4f} servings')
    total_cal = sum((calories[f] * servings_vars[f].X for f in foods))
    total_prot = sum((protein[f] * servings_vars[f].X for f in foods))
    total_fat = sum((fat[f] * servings_vars[f].X for f in foods))
    total_vitc = sum((vitamin_c[f] * servings_vars[f].X for f in foods))
    print('\n--- Achieved Nutrient Totals ---')
    print(f'Calories: {total_cal:.2f} kcal')
    print(f'Protein: {total_prot:.2f} g')
    print(f'Fat: {total_fat:.2f} g')
    print(f'Vitamin C: {total_vitc:.2f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')