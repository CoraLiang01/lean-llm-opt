import gurobipy as gp
import pandas as pd
import numpy as np

def solve_nutrition_problem():
    path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S2/Large-scale-or/Others_example/Others4/cost.csv'
    df = pd.read_csv(path, sep=',')
    foods = df['Food'].astype(str).tolist()
    calories = dict(zip(df['Food'].astype(str), df['Calories'].astype(float)))
    protein = dict(zip(df['Food'].astype(str), df['Protein(g)'].astype(float)))
    fat = dict(zip(df['Food'].astype(str), df['Fat(g)'].astype(float)))
    vitamin_c = dict(zip(df['Food'].astype(str), df['VitaminC(mg)'].astype(float)))
    cost = dict(zip(df['Food'].astype(str), df['Cost'].astype(float)))
    for f in foods:
        for param, d in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
            if f not in d or pd.isnull(d[f]):
                raise ValueError(f"Missing value for {param} in food '{f}'")
    m = gp.Model('OneDayMealPlan')
    x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[f] * x[f] for f in foods)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((calories[f] * x[f] for f in foods)) >= 2000, name='calories_min')
    m.addConstr(gp.quicksum((protein[f] * x[f] for f in foods)) >= 50, name='protein_min')
    m.addConstr(gp.quicksum((vitamin_c[f] * x[f] for f in foods)) >= 60, name='vitamin_c_min')
    m.addConstr(gp.quicksum((fat[f] * x[f] for f in foods)) <= 70, name='fat_max')
    m.optimize()
    return m
m = solve_nutrition_problem()