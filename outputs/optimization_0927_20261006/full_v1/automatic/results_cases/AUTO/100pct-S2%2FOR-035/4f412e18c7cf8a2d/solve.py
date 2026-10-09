import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
foods = df['Food'].tolist()

def to_numeric(series, colname):
    try:
        return pd.to_numeric(series, errors='raise')
    except Exception as e:
        raise ValueError(f"Column '{colname}' contains non-numeric or missing values.") from e
calories = dict(zip(foods, to_numeric(df['Calories'], 'Calories')))
protein = dict(zip(foods, to_numeric(df['Protein(g)'], 'Protein(g)')))
fat = dict(zip(foods, to_numeric(df['Fat(g)'], 'Fat(g)')))
vitamin_c = dict(zip(foods, to_numeric(df['VitaminC(mg)'], 'VitaminC(mg)')))
cost = dict(zip(foods, to_numeric(df['Cost'], 'Cost')))
for f in foods:
    for (param, d) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
        if f not in d:
            raise ValueError(f"Missing {param} data for food '{f}'.")
m = gp.Model('DietOptimization')
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
        val = servings_vars[f].X
        if val > 1e-05:
            print(f'{f}: {val:.3f} servings')
else:
    print(f'No optimal solution found. Status: {m.status}')