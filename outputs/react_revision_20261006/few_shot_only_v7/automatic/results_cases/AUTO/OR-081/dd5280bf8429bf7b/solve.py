import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_nutrition_problem():
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv'
    df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
    required_cols = ['Food', 'Calories', 'Protein(g)', 'Fat(g)', 'VitaminC(mg)', 'Cost']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f'Missing required column: {col}')
    foods = df['Food'].tolist()
    if len(set(foods)) != len(foods):
        raise ValueError("Duplicate food identifiers found in 'Food' column.")

    def to_float_series(series, colname):
        try:
            return series.astype(float)
        except Exception as e:
            raise ValueError(f"Column '{colname}' contains non-numeric values.") from e
    calories = dict(zip(foods, to_float_series(df['Calories'], 'Calories')))
    protein = dict(zip(foods, to_float_series(df['Protein(g)'], 'Protein(g)')))
    fat = dict(zip(foods, to_float_series(df['Fat(g)'], 'Fat(g)')))
    vitamin_c = dict(zip(foods, to_float_series(df['VitaminC(mg)'], 'VitaminC(mg)')))
    cost = dict(zip(foods, to_float_series(df['Cost'], 'Cost')))
    for f in foods:
        for (d, name) in [(calories, 'Calories'), (protein, 'Protein(g)'), (fat, 'Fat(g)'), (vitamin_c, 'VitaminC(mg)'), (cost, 'Cost')]:
            if f not in d:
                raise KeyError(f"Missing {name} for food '{f}'.")
    m = gp.Model('OneDayMealPlan')
    servings_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[f] * servings_vars[f] for f in foods)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((calories[f] * servings_vars[f] for f in foods)) >= 2000, name='cal_min')
    m.addConstr(gp.quicksum((protein[f] * servings_vars[f] for f in foods)) >= 50, name='prot_min')
    m.addConstr(gp.quicksum((vitamin_c[f] * servings_vars[f] for f in foods)) >= 60, name='vitc_min')
    m.addConstr(gp.quicksum((fat[f] * servings_vars[f] for f in foods)) <= 70, name='fat_max')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'Optimal total value/cost: {m.objVal:.6f}')
        for f in foods:
            print(f'{servings_vars[f].VarName} {servings_vars[f].X:.6f}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_nutrition_problem()