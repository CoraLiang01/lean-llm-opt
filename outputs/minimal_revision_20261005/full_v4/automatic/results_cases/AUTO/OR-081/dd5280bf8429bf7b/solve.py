import gurobipy as gp
import pandas as pd
import numpy as np
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv', sep=',')
foods = cost_df['Food'].astype(str).tolist()
calories = dict(zip(cost_df['Food'].astype(str), cost_df['Calories'].astype(float)))
protein = dict(zip(cost_df['Food'].astype(str), cost_df['Protein(g)'].astype(float)))
fat = dict(zip(cost_df['Food'].astype(str), cost_df['Fat(g)'].astype(float)))
vitamin_c = dict(zip(cost_df['Food'].astype(str), cost_df['VitaminC(mg)'].astype(float)))
cost = dict(zip(cost_df['Food'].astype(str), cost_df['Cost'].astype(float)))
for f in foods:
    if f not in calories or f not in protein or f not in fat or (f not in vitamin_c) or (f not in cost):
        raise ValueError(f'Missing parameter(s) for food: {f}')
m = gp.Model('MealPlan')
x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * x[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * x[f] for f in foods)) >= 2000, name='calories')
m.addConstr(gp.quicksum((protein[f] * x[f] for f in foods)) >= 50, name='protein')
m.addConstr(gp.quicksum((vitamin_c[f] * x[f] for f in foods)) >= 60, name='vitamin_c')
m.addConstr(gp.quicksum((fat[f] * x[f] for f in foods)) <= 70, name='fat')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for f in foods:
        var = x[f]
        print(f'{var.VarName} {var.X:.6f}')
else:
    print(f'Solver status: {m.status}')