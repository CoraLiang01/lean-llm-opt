import gurobipy as gp
import pandas as pd
import numpy as np
import math
import re

def solve_nutrition_blending():
    path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Others_example/Others4/cost.csv'
    df = pd.read_csv(path, sep=',')
    foods = df['Food'].astype(str).tolist()

    def get_col_dict(colname, dtype=float):
        if colname not in df.columns:
            raise KeyError(f"Required column '{colname}' not found in CSV.")
        vals = df[colname]
        if dtype == float:
            vals = vals.astype(float)
        elif dtype == int:
            vals = vals.astype(int)
        return dict(zip(df['Food'].astype(str), vals))
    cost = get_col_dict('Cost', float)
    calories = get_col_dict('Calories', float)
    protein = get_col_dict('Protein(g)', float)
    fat = get_col_dict('Fat(g)', float)
    vitamin_c = get_col_dict('VitaminC(mg)', float)
    for food in foods:
        for (dct, name) in [(cost, 'Cost'), (calories, 'Calories'), (protein, 'Protein(g)'), (fat, 'Fat(g)'), (vitamin_c, 'VitaminC(mg)')]:
            if food not in dct or pd.isnull(dct[food]):
                raise ValueError(f"Missing or NaN value for '{name}' in food '{food}'.")
    m = gp.Model('DietBlending')
    m.Params.MIPGap = 0.0001
    x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[f] * x[f] for f in foods)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((calories[f] * x[f] for f in foods)) >= 2000, name='calories_min')
    m.addConstr(gp.quicksum((protein[f] * x[f] for f in foods)) >= 50, name='protein_min')
    m.addConstr(gp.quicksum((vitamin_c[f] * x[f] for f in foods)) >= 60, name='vitaminC_min')
    m.addConstr(gp.quicksum((fat[f] * x[f] for f in foods)) <= 70, name='fat_max')
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal:.6f}')
        for f in foods:
            var = x[f]
            print(f'{var.VarName} {var.X:.6f}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_nutrition_blending()