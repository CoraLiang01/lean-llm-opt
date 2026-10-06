import gurobipy as gp
import pandas as pd
import numpy as np

def solve_nutrition_problem():
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Others_example/Others4/cost.csv'
    df = pd.read_csv(cost_path, sep=',')
    foods = df['Food'].astype(str).tolist()
    n_foods = len(foods)
    calories = dict(zip(df['Food'].astype(str), df['Calories'].astype(float)))
    protein = dict(zip(df['Food'].astype(str), df['Protein(g)'].astype(float)))
    fat = dict(zip(df['Food'].astype(str), df['Fat(g)'].astype(float)))
    vitamin_c = dict(zip(df['Food'].astype(str), df['VitaminC(mg)'].astype(float)))
    cost = dict(zip(df['Food'].astype(str), df['Cost'].astype(float)))
    for f in foods:
        if f not in calories or f not in protein or f not in fat or (f not in vitamin_c) or (f not in cost):
            raise ValueError(f'Missing nutrient or cost data for food: {f}')
    m = gp.Model('DietProblem')
    x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[f] * x[f] for f in foods)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((calories[f] * x[f] for f in foods)) >= 2000, name='calories_min')
    m.addConstr(gp.quicksum((protein[f] * x[f] for f in foods)) >= 50, name='protein_min')
    m.addConstr(gp.quicksum((vitamin_c[f] * x[f] for f in foods)) >= 60, name='vitaminc_min')
    m.addConstr(gp.quicksum((fat[f] * x[f] for f in foods)) <= 70, name='fat_max')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal:.6f}')
        for f in foods:
            var = x[f]
            print(f'{var.VarName} {var.X:.6f}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_nutrition_problem()