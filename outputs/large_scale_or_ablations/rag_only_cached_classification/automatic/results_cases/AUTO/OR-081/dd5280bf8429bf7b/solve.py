import pandas as pd
import numpy as np
from gurobipy import Model, GRB
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv', sep=',')
cost_df['Food_norm'] = cost_df['Food'].astype(str).str.strip().str.casefold()
foods = cost_df['Food'].tolist()
calories = dict(zip(foods, cost_df['Calories'].astype(float)))
protein = dict(zip(foods, cost_df['Protein(g)'].astype(float)))
fat = dict(zip(foods, cost_df['Fat(g)'].astype(float)))
vitamin_c = dict(zip(foods, cost_df['VitaminC(mg)'].astype(float)))
cost = dict(zip(foods, cost_df['Cost'].astype(float)))
for f in foods:
    if f not in calories or f not in protein or f not in fat or (f not in vitamin_c) or (f not in cost):
        raise ValueError(f'Missing parameter(s) for food: {f}')
m = Model('meal_plan')
x = m.addVars(foods, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(sum((cost[f] * x[f] for f in foods)), GRB.MINIMIZE)
m.addConstr(sum((calories[f] * x[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(sum((protein[f] * x[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(sum((vitamin_c[f] * x[f] for f in foods)) >= 60, name='vitaminC_min')
m.addConstr(sum((fat[f] * x[f] for f in foods)) <= 70, name='fat_max')
m.optimize()