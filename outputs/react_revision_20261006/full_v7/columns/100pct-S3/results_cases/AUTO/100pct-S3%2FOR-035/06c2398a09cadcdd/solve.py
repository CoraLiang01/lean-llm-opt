import gurobipy as gp
import pandas as pd
import numpy as np

def solve_nutrition_problem():
    df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Others_example/Others4/cost.csv', dtype=str, keep_default_na=False)
    foods = df['Food'].tolist()

    def get_numeric_series(colname, required=True):
        if colname not in df.columns:
            raise KeyError(f"Required column '{colname}' not found in CSV.")
        s = df[colname].copy()
        if required and (s == '').any():
            raise ValueError(f"Missing value(s) in required column '{colname}'.")
        try:
            return s.astype(float)
        except Exception as e:
            raise ValueError(f"Could not convert column '{colname}' to float: {e}")
    cost = dict(zip(foods, get_numeric_series('Cost')))
    calories = dict(zip(foods, get_numeric_series('Calories')))
    protein = dict(zip(foods, get_numeric_series('Protein(g)')))
    fat = dict(zip(foods, get_numeric_series('Fat(g)')))
    vitamin_c = dict(zip(foods, get_numeric_series('VitaminC(mg)')))
    for f in foods:
        for (param, d) in [('Cost', cost), ('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c)]:
            if f not in d:
                raise ValueError(f"Missing {param} for food '{f}'.")
    m = gp.Model('DietProblem')
    servings_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[f] * servings_vars[f] for f in foods)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((calories[f] * servings_vars[f] for f in foods)) >= 2000, name='calories')
    m.addConstr(gp.quicksum((protein[f] * servings_vars[f] for f in foods)) >= 50, name='protein')
    m.addConstr(gp.quicksum((vitamin_c[f] * servings_vars[f] for f in foods)) >= 60, name='vitamin_c')
    m.addConstr(gp.quicksum((fat[f] * servings_vars[f] for f in foods)) <= 70, name='fat')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal:.6f}')
        for f in foods:
            var = servings_vars[f]
            print(f'{var.VarName} {var.X:.6f}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_nutrition_problem()