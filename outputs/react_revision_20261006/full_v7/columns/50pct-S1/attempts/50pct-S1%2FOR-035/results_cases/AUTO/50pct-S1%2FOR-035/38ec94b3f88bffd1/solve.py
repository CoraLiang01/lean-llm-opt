import gurobipy as gp
import pandas as pd
import numpy as np

def solve_nutrition_problem():
    df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Others_example/Others4/cost.csv', dtype=str, keep_default_na=False)
    foods = df['Food'].tolist()

    def get_numeric_col(colname, dtype):
        vals = df[colname].apply(lambda x: x.strip() if isinstance(x, str) else x)
        try:
            if dtype == float:
                return dict(zip(foods, vals.astype(float)))
            elif dtype == int:
                return dict(zip(foods, vals.astype(int)))
            else:
                raise ValueError('Unsupported dtype')
        except Exception as e:
            raise ValueError(f"Column '{colname}' could not be converted to {dtype}: {e}")
    calories = get_numeric_col('Calories', int)
    protein = get_numeric_col('Protein(g)', float)
    fat = get_numeric_col('Fat(g)', float)
    vitamin_c = get_numeric_col('VitaminC(mg)', float)
    cost = get_numeric_col('Cost', float)
    for food in foods:
        for (param, param_dict) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
            if food not in param_dict:
                raise ValueError(f"Missing {param} value for food '{food}'")
    m = gp.Model('DietProblem')
    servings_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[food] * servings_vars[food] for food in foods)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((calories[food] * servings_vars[food] for food in foods)) >= 2000, name='calories')
    m.addConstr(gp.quicksum((protein[food] * servings_vars[food] for food in foods)) >= 50, name='protein')
    m.addConstr(gp.quicksum((vitamin_c[food] * servings_vars[food] for food in foods)) >= 60, name='vitamin_c')
    m.addConstr(gp.quicksum((fat[food] * servings_vars[food] for food in foods)) <= 70, name='fat')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal:.6f}')
        for food in foods:
            var = servings_vars[food]
            print(f'{var.VarName} {var.X:.6f}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_nutrition_problem()