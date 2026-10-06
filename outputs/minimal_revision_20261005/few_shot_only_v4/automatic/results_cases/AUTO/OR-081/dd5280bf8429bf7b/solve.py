import gurobipy as gp
import pandas as pd
import numpy as np
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv', sep=',')
required_cols = ['Food', 'Calories', 'Protein(g)', 'Fat(g)', 'VitaminC(mg)', 'Cost']
for col in required_cols:
    if col not in cost_df.columns:
        raise KeyError(f'Missing required column: {col}')
foods = list(cost_df['Food'])
calories = dict(zip(cost_df['Food'], cost_df['Calories']))
protein = dict(zip(cost_df['Food'], cost_df['Protein(g)']))
fat = dict(zip(cost_df['Food'], cost_df['Fat(g)']))
vitaminc = dict(zip(cost_df['Food'], cost_df['VitaminC(mg)']))
cost = dict(zip(cost_df['Food'], cost_df['Cost']))
for f in foods:
    for (d, name) in [(calories, 'Calories'), (protein, 'Protein(g)'), (fat, 'Fat(g)'), (vitaminc, 'VitaminC(mg)'), (cost, 'Cost')]:
        if f not in d or pd.isnull(d[f]):
            raise ValueError(f"Missing or NaN value for {name} in food '{f}'")
m = gp.Model('MealPlan')
x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * x[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * x[f] for f in foods)) >= 2000, name='calories')
m.addConstr(gp.quicksum((protein[f] * x[f] for f in foods)) >= 50, name='protein')
m.addConstr(gp.quicksum((vitaminc[f] * x[f] for f in foods)) >= 60, name='vitaminc')
m.addConstr(gp.quicksum((fat[f] * x[f] for f in foods)) <= 70, name='fat')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for f in foods:
        print(f'{x[f].VarName}: {x[f].X:.6f}')
else:
    print(f'Solver status: {m.status}')