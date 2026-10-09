import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
foods = df['Food'].astype(str).tolist()

def get_numeric_series(df, colname):
    candidates = [c for c in df.columns if c.strip().casefold() == colname.strip().casefold()]
    if not candidates:
        raise KeyError(f"Column '{colname}' not found in CSV columns: {list(df.columns)}")
    col = candidates[0]
    return pd.to_numeric(df[col], errors='raise')
calories = get_numeric_series(df, 'Calories').to_numpy()
protein = get_numeric_series(df, 'Protein(g)').to_numpy()
fat = get_numeric_series(df, 'Fat(g)').to_numpy()
vitamin_c = get_numeric_series(df, 'VitaminC(mg)').to_numpy()
cost = get_numeric_series(df, 'Cost').to_numpy()
food_idx = {food: i for (i, food) in enumerate(foods)}
m = gp.Model('DietOptimization')
x_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[food_idx[food]] * x_vars[food] for food in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[food_idx[food]] * x_vars[food] for food in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[food_idx[food]] * x_vars[food] for food in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[food_idx[food]] * x_vars[food] for food in foods)) >= 60, name='vitaminC_min')
m.addConstr(gp.quicksum((fat[food_idx[food]] * x_vars[food] for food in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for food in foods:
        servings = x_vars[food].X
        if servings > 1e-05:
            print(f'{food}: {servings:.3f} servings')
else:
    print(f'No optimal solution found. Status: {m.status}')