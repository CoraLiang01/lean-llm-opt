import gurobipy as gp
import pandas as pd
import numpy as np

def solve_nutrition_problem():
    df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Others_example/Others4/cost.csv', sep=',', dtype=str, keep_default_na=False)
    foods = df['Food'].tolist()
    required_cols = {'Calories': 'Calories', 'Protein(g)': 'Protein(g)', 'Fat(g)': 'Fat(g)', 'VitaminC(mg)': 'VitaminC(mg)', 'Cost': 'Cost'}
    for col in required_cols.values():
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in CSV.")

    def to_numeric(series, colname):
        try:
            return pd.to_numeric(series, errors='raise')
        except Exception as e:
            raise ValueError(f"Column '{colname}' contains non-numeric values.") from e
    calories = dict(zip(foods, to_numeric(df[required_cols['Calories']], 'Calories')))
    protein = dict(zip(foods, to_numeric(df[required_cols['Protein(g)']], 'Protein(g)')))
    fat = dict(zip(foods, to_numeric(df[required_cols['Fat(g)']], 'Fat(g)')))
    vitamin_c = dict(zip(foods, to_numeric(df[required_cols['VitaminC(mg)']], 'VitaminC(mg)')))
    cost = dict(zip(foods, to_numeric(df[required_cols['Cost']], 'Cost')))
    for f in foods:
        for (param, d) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
            if f not in d:
                raise KeyError(f"Missing {param} for food '{f}'.")
    m = gp.Model('OneDayMealPlan')
    servings_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[f] * servings_vars[f] for f in foods)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((calories[f] * servings_vars[f] for f in foods)) >= 2000, name='calories_min')
    m.addConstr(gp.quicksum((protein[f] * servings_vars[f] for f in foods)) >= 50, name='protein_min')
    m.addConstr(gp.quicksum((vitamin_c[f] * servings_vars[f] for f in foods)) >= 60, name='vitaminc_min')
    m.addConstr(gp.quicksum((fat[f] * servings_vars[f] for f in foods)) <= 70, name='fat_max')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == gp.GRB.OPTIMAL:
        print(f'Optimal total value/cost: {m.ObjVal:.4f}')
        for f in foods:
            var = servings_vars[f]
            print(f'{var.VarName} {var.X:.6f}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_nutrition_problem()