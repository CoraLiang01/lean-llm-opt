import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
foods = df['Food'].tolist()

def to_float_series(df, col):
    return df[col].astype(str).str.strip().replace('', '0').astype(float)
calories = dict(zip(foods, to_float_series(df, 'Calories')))
protein = dict(zip(foods, to_float_series(df, 'Protein(g)')))
fat = dict(zip(foods, to_float_series(df, 'Fat(g)')))
vitamin_c = dict(zip(foods, to_float_series(df, 'VitaminC(mg)')))
cost = dict(zip(foods, to_float_series(df, 'Cost')))
for food in foods:
    for (param, dct) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
        if food not in dct:
            raise ValueError(f"Missing {param} data for food '{food}'.")
m = gp.Model('DietProblem')
x_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[food] * x_vars[food] for food in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[food] * x_vars[food] for food in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[food] * x_vars[food] for food in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[food] * x_vars[food] for food in foods)) >= 60, name='vitamin_c_min')
m.addConstr(gp.quicksum((fat[food] * x_vars[food] for food in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: ${m.objVal:.2f}')
    print('Optimal servings per food (nonzero only):')
    for food in foods:
        servings = x_vars[food].X
        if servings > 1e-06:
            print(f'  {food}: {servings:.4f} servings')
else:
    print(f'No optimal solution found. Status: {m.status}')