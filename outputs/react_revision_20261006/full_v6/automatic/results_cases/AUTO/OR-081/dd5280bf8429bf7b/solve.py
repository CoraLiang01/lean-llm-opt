import gurobipy as gp
import pandas as pd
import numpy as np
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv', sep=',', dtype=str, keep_default_na=False)
foods = cost_df['Food'].astype(str).tolist()
for col in ['Calories', 'Protein(g)', 'Fat(g)', 'VitaminC(mg)', 'Cost']:
    cost_df[col] = pd.to_numeric(cost_df[col], errors='raise')
calories = dict(zip(cost_df['Food'], cost_df['Calories']))
protein = dict(zip(cost_df['Food'], cost_df['Protein(g)']))
fat = dict(zip(cost_df['Food'], cost_df['Fat(g)']))
vitamin_c = dict(zip(cost_df['Food'], cost_df['VitaminC(mg)']))
cost = dict(zip(cost_df['Food'], cost_df['Cost']))
for f in foods:
    for (param, d) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
        if f not in d:
            raise ValueError(f"Missing {param} for food '{f}'")
m = gp.Model('MealPlan')
servings_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * servings_vars[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * servings_vars[f] for f in foods)) >= 2000, name='cal_min')
m.addConstr(gp.quicksum((protein[f] * servings_vars[f] for f in foods)) >= 50, name='prot_min')
m.addConstr(gp.quicksum((vitamin_c[f] * servings_vars[f] for f in foods)) >= 60, name='vitc_min')
m.addConstr(gp.quicksum((fat[f] * servings_vars[f] for f in foods)) <= 70, name='fat_max')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.4f}')
    for f in foods:
        var = servings_vars[f]
        print(f'{var.VarName}: {var.X:.6f}')
else:
    print(f'Solver status: {m.status}')