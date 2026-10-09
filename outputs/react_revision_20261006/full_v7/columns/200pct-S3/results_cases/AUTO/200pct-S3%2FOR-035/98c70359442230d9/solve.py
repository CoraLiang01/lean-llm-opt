import gurobipy as gp
import pandas as pd
import numpy as np

def solve_nutrition_lp():
    df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Others_example/Others4/cost.csv', sep=',', dtype=str, keep_default_na=False)
    foods = df['Food'].tolist()
    if len(set(foods)) != len(foods):
        raise ValueError('Duplicate Food identifiers found in cost.csv.')

    def to_float_series(col):
        try:
            return df[col].astype(float)
        except Exception as e:
            raise ValueError(f"Column '{col}' could not be fully converted to float: {e}")
    calories = dict(zip(foods, to_float_series('Calories')))
    protein = dict(zip(foods, to_float_series('Protein(g)')))
    fat = dict(zip(foods, to_float_series('Fat(g)')))
    vitamin_c = dict(zip(foods, to_float_series('VitaminC(mg)')))
    cost = dict(zip(foods, to_float_series('Cost')))
    for food in foods:
        for (param, d) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
            if food not in d:
                raise ValueError(f"Missing {param} data for food '{food}'.")
    m = gp.Model('DietProblem')
    servings_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[f] * servings_vars[f] for f in foods)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((calories[f] * servings_vars[f] for f in foods)) >= 2000, name='calories')
    m.addConstr(gp.quicksum((protein[f] * servings_vars[f] for f in foods)) >= 50, name='protein')
    m.addConstr(gp.quicksum((vitamin_c[f] * servings_vars[f] for f in foods)) >= 60, name='vitaminC')
    m.addConstr(gp.quicksum((fat[f] * servings_vars[f] for f in foods)) <= 70, name='fat')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == gp.GRB.OPTIMAL:
        print(f'Optimal total value/cost: {m.ObjVal:.6f}')
        for f in foods:
            print(f'{servings_vars[f].VarName} {servings_vars[f].X:.6f}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_nutrition_lp()