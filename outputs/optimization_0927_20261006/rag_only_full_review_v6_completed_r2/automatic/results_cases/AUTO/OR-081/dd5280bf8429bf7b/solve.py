import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv', dtype=str, keep_default_na=False)
cost_df['Food'] = cost_df['Food'].str.strip()
for col in ['Calories', 'Protein(g)', 'Fat(g)', 'VitaminC(mg)', 'Cost']:
    cost_df[col] = cost_df[col].astype(float)
foods = cost_df['Food'].tolist()
calories = dict(zip(cost_df['Food'], cost_df['Calories']))
protein = dict(zip(cost_df['Food'], cost_df['Protein(g)']))
fat = dict(zip(cost_df['Food'], cost_df['Fat(g)']))
vitamin_c = dict(zip(cost_df['Food'], cost_df['VitaminC(mg)']))
cost = dict(zip(cost_df['Food'], cost_df['Cost']))
for f in foods:
    if f not in calories or f not in protein or f not in fat or (f not in vitamin_c) or (f not in cost):
        raise ValueError(f'Missing parameter(s) for food: {f}')
m = Model('meal_plan')
servings_vars = m.addVars(foods, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(quicksum((cost[f] * servings_vars[f] for f in foods)), GRB.MINIMIZE)
m.addConstr(quicksum((calories[f] * servings_vars[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(quicksum((protein[f] * servings_vars[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(quicksum((vitamin_c[f] * servings_vars[f] for f in foods)) >= 60, name='vitaminC_min')
m.addConstr(quicksum((fat[f] * servings_vars[f] for f in foods)) <= 70, name='fat_max')
m.optimize()