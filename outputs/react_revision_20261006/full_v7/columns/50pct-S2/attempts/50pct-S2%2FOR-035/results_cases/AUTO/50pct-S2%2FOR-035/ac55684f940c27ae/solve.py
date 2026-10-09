import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
foods = df['Food'].tolist()
required_columns = {'Calories': 'Calories', 'Protein(g)': 'Protein(g)', 'Fat(g)': 'Fat(g)', 'VitaminC(mg)': 'VitaminC(mg)', 'Cost': 'Cost'}
for col in required_columns.values():
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in cost.csv.")

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
for food in foods:
    for (param, param_dict) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
        if food not in param_dict:
            raise KeyError(f"Missing {param} value for food '{food}'.")

def solve_meal_plan(foods, calories, protein, fat, vitamin_c, cost):
    m = gp.Model('MealPlanDiet')
    servings_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[f] * servings_vars[f] for f in foods)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((calories[f] * servings_vars[f] for f in foods)) >= 2000, name='calories')
    m.addConstr(gp.quicksum((protein[f] * servings_vars[f] for f in foods)) >= 50, name='protein')
    m.addConstr(gp.quicksum((vitamin_c[f] * servings_vars[f] for f in foods)) >= 60, name='vitaminC')
    m.addConstr(gp.quicksum((fat[f] * servings_vars[f] for f in foods)) <= 70, name='fat')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_meal_plan(foods, calories, protein, fat, vitamin_c, cost)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X:.6f}')
else:
    print(f'Solver status: {m.status}')