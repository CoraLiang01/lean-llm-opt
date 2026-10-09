import gurobipy as gp
import pandas as pd
import numpy as np

def solve_nutrition_problem():
    df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Others_example/Others4/cost.csv', dtype=str, keep_default_na=False)
    foods = df['Food'].tolist()
    required_columns = {'Calories': 'Calories', 'Protein(g)': 'Protein(g)', 'Fat(g)': 'Fat(g)', 'VitaminC(mg)': 'VitaminC(mg)', 'Cost': 'Cost'}
    for col in required_columns.values():
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in CSV.")

    def to_numeric_series(series, colname):
        try:
            return pd.to_numeric(series, errors='raise')
        except Exception as e:
            raise ValueError(f"Column '{colname}' contains non-numeric values.") from e
    calories = dict(zip(foods, to_numeric_series(df[required_columns['Calories']], 'Calories')))
    protein = dict(zip(foods, to_numeric_series(df[required_columns['Protein(g)']], 'Protein(g)')))
    fat = dict(zip(foods, to_numeric_series(df[required_columns['Fat(g)']], 'Fat(g)')))
    vitamin_c = dict(zip(foods, to_numeric_series(df[required_columns['VitaminC(mg)']], 'VitaminC(mg)')))
    cost = dict(zip(foods, to_numeric_series(df[required_columns['Cost']], 'Cost')))
    for f in foods:
        for (param, d) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
            if f not in d:
                raise ValueError(f"Missing {param} value for food '{f}'.")
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
        print(f'Optimal total value/cost: {m.objVal:.4f}')
        for f in foods:
            var = servings_vars[f]
            print(f'{var.VarName} {var.X:.6f}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_nutrition_problem()