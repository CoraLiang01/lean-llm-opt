import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',')
foods = df['Food'].astype(str).tolist()

def extract_param(colname, dtype=float):
    if df[colname].isnull().any():
        raise ValueError(f"Missing values in column '{colname}'")
    return dict(zip(df['Food'].astype(str), df[colname].astype(dtype)))
calories = extract_param('Calories', float)
protein = extract_param('Protein(g)', float)
fat = extract_param('Fat(g)', float)
vitamin_c = extract_param('VitaminC(mg)', float)
cost = extract_param('Cost', float)
for food in foods:
    for (param, d) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
        if food not in d:
            raise ValueError(f"Missing {param} for food '{food}'")

def solve_nutrition_lp(foods, calories, protein, fat, vitamin_c, cost):
    m = gp.Model('DietBlending')
    x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i] * x[i] for i in foods)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((calories[i] * x[i] for i in foods)) >= 2000, name='calories')
    m.addConstr(gp.quicksum((protein[i] * x[i] for i in foods)) >= 50, name='protein')
    m.addConstr(gp.quicksum((vitamin_c[i] * x[i] for i in foods)) >= 60, name='vitaminC')
    m.addConstr(gp.quicksum((fat[i] * x[i] for i in foods)) <= 70, name='fat')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_nutrition_lp(foods, calories, protein, fat, vitamin_c, cost)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X:.6f}')
else:
    print(f'Solver status: {m.status}')