import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem():
    df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Others_example/Others4/cost.csv', sep=',', dtype=str, keep_default_na=False)
    foods = df['Food'].tolist()

    def to_float_series(series, colname):
        try:
            vals = pd.to_numeric(series, errors='raise')
        except Exception as e:
            raise ValueError(f"Column '{colname}' contains non-numeric or missing values.") from e
        if len(vals) != len(foods):
            raise ValueError(f"Length mismatch in column '{colname}'.")
        return vals.values
    calories = to_float_series(df['Calories'], 'Calories')
    protein = to_float_series(df['Protein(g)'], 'Protein(g)')
    fat = to_float_series(df['Fat(g)'], 'Fat(g)')
    vitamin_c = to_float_series(df['VitaminC(mg)'], 'VitaminC(mg)')
    cost = to_float_series(df['Cost'], 'Cost')
    calories_dict = dict(zip(foods, calories))
    protein_dict = dict(zip(foods, protein))
    fat_dict = dict(zip(foods, fat))
    vitamin_c_dict = dict(zip(foods, vitamin_c))
    cost_dict = dict(zip(foods, cost))
    for food in foods:
        for (d, name) in [(calories_dict, 'Calories'), (protein_dict, 'Protein(g)'), (fat_dict, 'Fat(g)'), (vitamin_c_dict, 'VitaminC(mg)'), (cost_dict, 'Cost')]:
            if food not in d:
                raise ValueError(f"Missing {name} for food '{food}'.")
    m = gp.Model('DietPlan')
    m.Params.MIPGap = 0.0001
    servings_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost_dict[i] * servings_vars[i] for i in foods)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((calories_dict[i] * servings_vars[i] for i in foods)) >= 2000, name='calories_min')
    m.addConstr(gp.quicksum((protein_dict[i] * servings_vars[i] for i in foods)) >= 50, name='protein_min')
    m.addConstr(gp.quicksum((vitamin_c_dict[i] * servings_vars[i] for i in foods)) >= 60, name='vitaminc_min')
    m.addConstr(gp.quicksum((fat_dict[i] * servings_vars[i] for i in foods)) <= 70, name='fat_max')
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X:.6f}')
else:
    print(f'Solver status: {m.status}')