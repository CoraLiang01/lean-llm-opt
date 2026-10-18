import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(path, sep=',')
foods = df['Food'].tolist()
calories = dict(zip(df['Food'], df['Calories']))
protein = dict(zip(df['Food'], df['Protein(g)']))
fat = dict(zip(df['Food'], df['Fat(g)']))
vitC = dict(zip(df['Food'], df['VitaminC(mg)']))
cost = dict(zip(df['Food'], df['Cost']))
m = gp.Model('NutritionPlan')
x = m.addVars(foods)
m.setObjective(gp.quicksum((cost[i] * x[i] for i in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[i] * x[i] for i in foods)) >= 2000)
m.addConstr(gp.quicksum((protein[i] * x[i] for i in foods)) >= 50)
m.addConstr(gp.quicksum((vitC[i] * x[i] for i in foods)) >= 60)
m.addConstr(gp.quicksum((fat[i] * x[i] for i in foods)) <= 70)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.4f} USD')
    print('\nOptimal servings per food:')
    for i in foods:
        val = x[i].X
        if val > 1e-06:
            print(f'  {i}: {val:.4f} servings')
else:
    print(f'Optimization ended with status {m.status}')