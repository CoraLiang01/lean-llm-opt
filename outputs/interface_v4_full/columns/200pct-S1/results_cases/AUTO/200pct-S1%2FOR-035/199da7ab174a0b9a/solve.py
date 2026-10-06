import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S1/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',')
foods = df['Food'].astype(str).tolist()
calories = df.set_index('Food')['Calories'].astype(float).to_dict()
protein = df.set_index('Food')['Protein(g)'].astype(float).to_dict()
fat = df.set_index('Food')['Fat(g)'].astype(float).to_dict()
vitamin_c = df.set_index('Food')['VitaminC(mg)'].astype(float).to_dict()
cost = df.set_index('Food')['Cost'].astype(float).to_dict()
for food in foods:
    if food not in calories or food not in protein or food not in fat or (food not in vitamin_c) or (food not in cost):
        raise ValueError(f'Missing nutrient or cost data for food: {food}')
m = gp.Model('DietProblem')
x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i] * x[i] for i in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[i] * x[i] for i in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[i] * x[i] for i in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[i] * x[i] for i in foods)) >= 60, name='vitamin_c_min')
m.addConstr(gp.quicksum((fat[i] * x[i] for i in foods)) <= 70, name='fat_max')
m.optimize()