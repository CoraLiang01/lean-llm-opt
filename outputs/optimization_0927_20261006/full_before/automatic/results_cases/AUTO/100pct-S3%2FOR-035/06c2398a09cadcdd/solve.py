import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',')
foods = df['Food'].astype(str).tolist()
calories = dict(zip(df['Food'].astype(str), df['Calories'].astype(float)))
protein = dict(zip(df['Food'].astype(str), df['Protein(g)'].astype(float)))
fat = dict(zip(df['Food'].astype(str), df['Fat(g)'].astype(float)))
vitamin_c = dict(zip(df['Food'].astype(str), df['VitaminC(mg)'].astype(float)))
cost = dict(zip(df['Food'].astype(str), df['Cost'].astype(float)))
m = gp.Model('DietProblem')
x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * x[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * x[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[f] * x[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[f] * x[f] for f in foods)) >= 60, name='vitamin_c_min')
m.addConstr(gp.quicksum((fat[f] * x[f] for f in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for f in foods:
        servings = x[f].X
        if servings > 1e-05:
            print(f'{f}: {servings:.4f} servings')
else:
    print(f'No optimal solution found. Status: {m.status}')