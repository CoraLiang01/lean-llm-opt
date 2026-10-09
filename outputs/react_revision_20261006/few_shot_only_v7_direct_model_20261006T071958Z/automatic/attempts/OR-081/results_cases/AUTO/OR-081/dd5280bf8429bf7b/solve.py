import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
foods = df['Food'].tolist()

def to_float(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Column '{colname}' could not be fully converted to float. Error: {e}")
calories = dict(zip(foods, to_float(df['Calories'], 'Calories')))
protein = dict(zip(foods, to_float(df['Protein(g)'], 'Protein(g)')))
fat = dict(zip(foods, to_float(df['Fat(g)'], 'Fat(g)')))
vitamin_c = dict(zip(foods, to_float(df['VitaminC(mg)'], 'VitaminC(mg)')))
cost = dict(zip(foods, to_float(df['Cost'], 'Cost')))
for food in foods:
    for (param, dct) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
        if food not in dct:
            raise ValueError(f"Missing {param} data for food '{food}'.")

def solve_meal_plan():
    m = gp.Model('MealPlan')
    servings_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[f] * servings_vars[f] for f in foods)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((calories[f] * servings_vars[f] for f in foods)) >= 2000, name='calories')
    m.addConstr(gp.quicksum((protein[f] * servings_vars[f] for f in foods)) >= 50, name='protein')
    m.addConstr(gp.quicksum((vitamin_c[f] * servings_vars[f] for f in foods)) >= 60, name='vitaminC')
    m.addConstr(gp.quicksum((fat[f] * servings_vars[f] for f in foods)) <= 70, name='fat')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_meal_plan()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X:.6f}')
else:
    print(f'Solver status: {m.status}')