import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',')
expected_cols = ['Food', 'Calories', 'Protein(g)', 'Fat(g)', 'VitaminC(mg)', 'Cost']
for col in expected_cols:
    if col not in df.columns:
        raise KeyError(f'Missing required column: {col}')
foods = df['Food'].astype(str).tolist()
n_foods = len(foods)
calories = dict(zip(foods, df['Calories'].astype(float)))
protein = dict(zip(foods, df['Protein(g)'].astype(float)))
fat = dict(zip(foods, df['Fat(g)'].astype(float)))
vitaminc = dict(zip(foods, df['VitaminC(mg)'].astype(float)))
cost = dict(zip(foods, df['Cost'].astype(float)))
for food in foods:
    for (d, name) in [(calories, 'Calories'), (protein, 'Protein(g)'), (fat, 'Fat(g)'), (vitaminc, 'VitaminC(mg)'), (cost, 'Cost')]:
        if food not in d or pd.isnull(d[food]):
            raise ValueError(f"Missing or NaN value for {name} in food '{food}'")

def solve_nutrition_problem():
    m = gp.Model('DietProblem')
    x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[f] * x[f] for f in foods)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((calories[f] * x[f] for f in foods)) >= 2000, name='calories_min')
    m.addConstr(gp.quicksum((protein[f] * x[f] for f in foods)) >= 50, name='protein_min')
    m.addConstr(gp.quicksum((vitaminc[f] * x[f] for f in foods)) >= 60, name='vitaminc_min')
    m.addConstr(gp.quicksum((fat[f] * x[f] for f in foods)) <= 70, name='fat_max')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_nutrition_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal objective value: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X:.6f}')
else:
    print(f'Solver status: {m.status}')