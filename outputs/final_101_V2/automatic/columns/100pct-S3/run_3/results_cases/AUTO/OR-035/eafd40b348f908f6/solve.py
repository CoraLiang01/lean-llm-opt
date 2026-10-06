import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S3/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',')
required_cols = ['Food', 'Calories', 'Protein(g)', 'Fat(g)', 'VitaminC(mg)', 'Cost']
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in cost.csv.")
foods = df['Food'].astype(str).tolist()
calories = df.set_index('Food')['Calories'].astype(float).to_dict()
protein = df.set_index('Food')['Protein(g)'].astype(float).to_dict()
fat = df.set_index('Food')['Fat(g)'].astype(float).to_dict()
vitaminc = df.set_index('Food')['VitaminC(mg)'].astype(float).to_dict()
cost = df.set_index('Food')['Cost'].astype(float).to_dict()
for food in foods:
    for param, d in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitaminc), ('Cost', cost)]:
        if food not in d or pd.isnull(d[food]):
            raise ValueError(f"Missing value for {param} in food '{food}'.")
m = gp.Model('DietPlan')
x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[food] * x[food] for food in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[food] * x[food] for food in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[food] * x[food] for food in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitaminc[food] * x[food] for food in foods)) >= 60, name='vitaminc_min')
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