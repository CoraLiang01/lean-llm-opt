import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem():
    path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S3/Large-scale-or/Others_example/Others4/cost.csv'
    df = pd.read_csv(path, sep=',')
    foods = df['Food'].astype(str).tolist()
    calories = dict(zip(df['Food'].astype(str), df['Calories'].astype(float)))
    protein = dict(zip(df['Food'].astype(str), df['Protein(g)'].astype(float)))
    fat = dict(zip(df['Food'].astype(str), df['Fat(g)'].astype(float)))
    vitamin_c = dict(zip(df['Food'].astype(str), df['VitaminC(mg)'].astype(float)))
    cost = dict(zip(df['Food'].astype(str), df['Cost'].astype(float)))
    for f in foods:
        for dct, name in [(calories, 'Calories'), (protein, 'Protein(g)'), (fat, 'Fat(g)'), (vitamin_c, 'VitaminC(mg)'), (cost, 'Cost')]:
            if f not in dct or pd.isnull(dct[f]):
                raise ValueError(f"Missing or NaN value for {name} in food '{f}'.")
    m = gp.Model('DietProblem')
    x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[f] * x[f] for f in foods)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((calories[f] * x[f] for f in foods)) >= 2000, name='calories_min')
    m.addConstr(gp.quicksum((protein[f] * x[f] for f in foods)) >= 50, name='protein_min')
    m.addConstr(gp.quicksum((vitamin_c[f] * x[f] for f in foods)) >= 60, name='vitaminC_min')
    m.addConstr(gp.quicksum((fat[f] * x[f] for f in foods)) <= 70, name='fat_max')
    m.optimize()
    return m
m = solve_problem()