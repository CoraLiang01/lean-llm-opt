import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['Food', 'Calories', 'Protein(g)', 'Fat(g)', 'VitaminC(mg)', 'Cost']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in cost.csv")
foods = df['Food'].tolist()

def to_float_series(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Column '{colname}' could not be fully converted to float: {e}")
calories = dict(zip(foods, to_float_series(df['Calories'], 'Calories')))
protein = dict(zip(foods, to_float_series(df['Protein(g)'], 'Protein(g)')))
fat = dict(zip(foods, to_float_series(df['Fat(g)'], 'Fat(g)')))
vitamin_c = dict(zip(foods, to_float_series(df['VitaminC(mg)'], 'VitaminC(mg)')))
cost = dict(zip(foods, to_float_series(df['Cost'], 'Cost')))
for f in foods:
    for (param, d) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
        if f not in d:
            raise KeyError(f"Food '{f}' missing parameter '{param}'")
m = gp.Model('MealPlanMinCost')
x_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * x_vars[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * x_vars[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[f] * x_vars[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[f] * x_vars[f] for f in foods)) >= 60, name='vitamin_c_min')
m.addConstr(gp.quicksum((fat[f] * x_vars[f] for f in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for f in foods:
        val = x_vars[f].X
        if val > 1e-05:
            print(f'{f}: {val:.4f} servings')
else:
    print(f'No optimal solution found. Status: {m.status}')