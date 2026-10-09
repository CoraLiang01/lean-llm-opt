import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Others_example/Others4/cost.csv'
cost_df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
foods = cost_df['Food'].tolist()

def to_float(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Column '{colname}' could not be fully converted to float: {e}")
calories = dict(zip(foods, to_float(cost_df['Calories'], 'Calories')))
protein = dict(zip(foods, to_float(cost_df['Protein(g)'], 'Protein(g)')))
fat = dict(zip(foods, to_float(cost_df['Fat(g)'], 'Fat(g)')))
vitamin_c = dict(zip(foods, to_float(cost_df['VitaminC(mg)'], 'VitaminC(mg)')))
cost = dict(zip(foods, to_float(cost_df['Cost'], 'Cost')))
for f in foods:
    for (d, name) in [(calories, 'Calories'), (protein, 'Protein(g)'), (fat, 'Fat(g)'), (vitamin_c, 'VitaminC(mg)'), (cost, 'Cost')]:
        if f not in d:
            raise KeyError(f"Missing {name} data for food '{f}'.")
m = gp.Model('MealPlanMinCost')
servings_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * servings_vars[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * servings_vars[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[f] * servings_vars[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[f] * servings_vars[f] for f in foods)) >= 60, name='vitaminC_min')
m.addConstr(gp.quicksum((fat[f] * servings_vars[f] for f in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Optimal Meal Plan (servings per food) ---')
    for f in foods:
        x = servings_vars[f].X
        if x > 1e-05:
            print(f'{f}: {x:.3f} servings')
else:
    print(f'No optimal solution found. Status: {m.status}')