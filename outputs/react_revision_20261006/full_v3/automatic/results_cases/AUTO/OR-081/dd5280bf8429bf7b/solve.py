import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',')
foods = df['Food'].astype(str).tolist()
calories = df.set_index('Food')['Calories'].astype(float).to_dict()
protein = df.set_index('Food')['Protein(g)'].astype(float).to_dict()
fat = df.set_index('Food')['Fat(g)'].astype(float).to_dict()
vitaminc = df.set_index('Food')['VitaminC(mg)'].astype(float).to_dict()
cost = df.set_index('Food')['Cost'].astype(float).to_dict()
for f in foods:
    if f not in calories or f not in protein or f not in fat or (f not in vitaminc) or (f not in cost):
        raise ValueError(f'Missing parameter(s) for food: {f}')

def solve_nutrition_blending(foods, calories, protein, fat, vitaminc, cost):
    m = gp.Model('NutritionBlending')
    x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[f] * x[f] for f in foods)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((calories[f] * x[f] for f in foods)) >= 2000, name='cal_min')
    m.addConstr(gp.quicksum((protein[f] * x[f] for f in foods)) >= 50, name='prot_min')
    m.addConstr(gp.quicksum((vitaminc[f] * x[f] for f in foods)) >= 60, name='vitc_min')
    m.addConstr(gp.quicksum((fat[f] * x[f] for f in foods)) <= 70, name='fat_max')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_nutrition_blending(foods, calories, protein, fat, vitaminc, cost)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X:.6f}')
else:
    print(f'Solver status: {m.status}')