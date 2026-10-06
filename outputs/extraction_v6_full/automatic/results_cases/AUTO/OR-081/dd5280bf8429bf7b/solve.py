import gurobipy as gp
import pandas as pd
import numpy as np
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv', sep=',')
foods = cost_df['Food'].astype(str).tolist()
calories = dict(zip(foods, cost_df['Calories'].astype(float)))
protein = dict(zip(foods, cost_df['Protein(g)'].astype(float)))
fat = dict(zip(foods, cost_df['Fat(g)'].astype(float)))
vitamin_c = dict(zip(foods, cost_df['VitaminC(mg)'].astype(float)))
cost = dict(zip(foods, cost_df['Cost'].astype(float)))
for f in foods:
    if f not in calories or f not in protein or f not in fat or (f not in vitamin_c) or (f not in cost):
        raise ValueError(f'Missing parameter(s) for food: {f}')
m = gp.Model('MealPlanMinCost')
x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='servings')
m.setObjective(gp.quicksum((cost[f] * x[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * x[f] for f in foods)) >= 2000, name='Calories_min')
m.addConstr(gp.quicksum((protein[f] * x[f] for f in foods)) >= 50, name='Protein_min')
m.addConstr(gp.quicksum((vitamin_c[f] * x[f] for f in foods)) >= 60, name='VitaminC_min')
m.addConstr(gp.quicksum((fat[f] * x[f] for f in foods)) <= 70, name='Fat_max')
m.optimize()