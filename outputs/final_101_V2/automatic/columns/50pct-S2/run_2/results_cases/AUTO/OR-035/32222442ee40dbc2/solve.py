import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/50pct/S2/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',')
foods = df['Food'].astype(str).tolist()

def build_param_dict(colname, dtype=float):
    vals = df.set_index('Food')[colname]
    if vals.isnull().any():
        missing = vals[vals.isnull()].index.tolist()
        raise ValueError(f'Missing values for foods: {missing} in column {colname}')
    return vals.astype(dtype).to_dict()
calories = build_param_dict('Calories', float)
protein = build_param_dict('Protein(g)', float)
fat = build_param_dict('Fat(g)', float)
vitamin_c = build_param_dict('VitaminC(mg)', float)
cost = build_param_dict('Cost', float)
for food in foods:
    for pname, pdict in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
        if food not in pdict:
            raise ValueError(f"Food '{food}' missing parameter '{pname}'")
m = gp.Model('DietProblem')
x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[food] * x[food] for food in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[food] * x[food] for food in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[food] * x[food] for food in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[food] * x[food] for food in foods)) >= 60, name='vitamin_c_min')
m.addConstr(gp.quicksum((fat[food] * x[food] for food in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for food in foods:
        servings = x[food].X
        if servings > 1e-05:
            print(f'{food}: {servings:.4f} servings')
else:
    print(f'No optimal solution found. Status: {m.status}')