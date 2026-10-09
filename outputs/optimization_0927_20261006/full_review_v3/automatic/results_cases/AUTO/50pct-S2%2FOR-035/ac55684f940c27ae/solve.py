import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
foods = df['Food'].tolist()

def to_float_series(series, colname):
    try:
        return pd.to_numeric(series, errors='raise')
    except Exception as e:
        raise ValueError(f"Column '{colname}' contains non-numeric values or missing data: {e}")
calories = to_float_series(df['Calories'], 'Calories')
protein = to_float_series(df['Protein(g)'], 'Protein(g)')
fat = to_float_series(df['Fat(g)'], 'Fat(g)')
vitamin_c = to_float_series(df['VitaminC(mg)'], 'VitaminC(mg)')
cost = to_float_series(df['Cost'], 'Cost')
calories_dict = dict(zip(foods, calories))
protein_dict = dict(zip(foods, protein))
fat_dict = dict(zip(foods, fat))
vitamin_c_dict = dict(zip(foods, vitamin_c))
cost_dict = dict(zip(foods, cost))
m = gp.Model('DietOptimization')
x_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost_dict[i] * x_vars[i] for i in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories_dict[i] * x_vars[i] for i in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein_dict[i] * x_vars[i] for i in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c_dict[i] * x_vars[i] for i in foods)) >= 60, name='vitamin_c_min')
m.addConstr(gp.quicksum((fat_dict[i] * x_vars[i] for i in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for i in foods:
        val = x_vars[i].X
        if val > 1e-05:
            print(f'{i}: {val:.3f} servings')
else:
    print(f'No optimal solution found. Status: {m.status}')