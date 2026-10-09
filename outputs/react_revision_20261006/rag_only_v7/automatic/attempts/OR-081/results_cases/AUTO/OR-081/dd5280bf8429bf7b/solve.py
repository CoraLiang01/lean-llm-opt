import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv', dtype=str, keep_default_na=False)
foods = cost_df['Food'].astype(str).tolist()
required_numeric_cols = ['Calories', 'Protein(g)', 'Fat(g)', 'VitaminC(mg)', 'Cost']
for col in required_numeric_cols:
    if not set(cost_df[col].apply(lambda x: x.strip() != '')).issubset({True}):
        raise ValueError(f"Missing values detected in column '{col}'")
    try:
        cost_df[col] = cost_df[col].astype(float)
    except Exception as e:
        raise ValueError(f"Column '{col}' could not be converted to float: {e}")
calories = dict(zip(foods, cost_df['Calories']))
protein = dict(zip(foods, cost_df['Protein(g)']))
fat = dict(zip(foods, cost_df['Fat(g)']))
vitaminc = dict(zip(foods, cost_df['VitaminC(mg)']))
cost = dict(zip(foods, cost_df['Cost']))
for f in foods:
    for (param, d) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitaminc), ('Cost', cost)]:
        if f not in d:
            raise ValueError(f"Missing {param} for food '{f}'")
m = gp.Model('meal_plan')
servings_vars = m.addVars(foods, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * servings_vars[f] for f in foods)), GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * servings_vars[f] for f in foods)) >= 2000, name='cal_min')
m.addConstr(gp.quicksum((protein[f] * servings_vars[f] for f in foods)) >= 50, name='prot_min')
m.addConstr(gp.quicksum((vitaminc[f] * servings_vars[f] for f in foods)) >= 60, name='vitc_min')
m.addConstr(gp.quicksum((fat[f] * servings_vars[f] for f in foods)) <= 70, name='fat_max')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'Optimal objective value: {m.ObjVal}')
    for f in foods:
        var = servings_vars[f]
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.Status}')