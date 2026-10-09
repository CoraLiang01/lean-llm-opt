import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',')
foods = df['Food'].astype(str).tolist()
n_foods = len(foods)
calories = dict(zip(foods, df['Calories'].astype(float)))
protein = dict(zip(foods, df['Protein(g)'].astype(float)))
fat = dict(zip(foods, df['Fat(g)'].astype(float)))
vitamin_c = dict(zip(foods, df['VitaminC(mg)'].astype(float)))
cost = dict(zip(foods, df['Cost'].astype(float)))
for f in foods:
    if f not in calories or f not in protein or f not in fat or (f not in vitamin_c) or (f not in cost):
        raise ValueError(f'Missing parameter(s) for food: {f}')
m = gp.Model('MealPlan')
x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * x[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * x[f] for f in foods)) >= 2000, name='cal_min')
m.addConstr(gp.quicksum((protein[f] * x[f] for f in foods)) >= 50, name='prot_min')
m.addConstr(gp.quicksum((vitamin_c[f] * x[f] for f in foods)) >= 60, name='vitc_min')
m.addConstr(gp.quicksum((fat[f] * x[f] for f in foods)) <= 70, name='fat_max')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.4f}')
    for f in foods:
        var = x[f]
        print(f'{var.VarName}: {var.X:.6f}')
else:
    print(f'Solver status: {m.status}')