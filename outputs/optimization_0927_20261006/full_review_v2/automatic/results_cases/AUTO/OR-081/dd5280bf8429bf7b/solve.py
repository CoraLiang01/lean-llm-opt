import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv'
cost_df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
foods = cost_df['Food'].tolist()

def to_numeric(series, colname):
    try:
        return pd.to_numeric(series, errors='raise')
    except Exception as e:
        raise ValueError(f"Column '{colname}' could not be fully converted to numeric: {e}")
calories = dict(zip(foods, to_numeric(cost_df['Calories'], 'Calories')))
protein = dict(zip(foods, to_numeric(cost_df['Protein(g)'], 'Protein(g)')))
fat = dict(zip(foods, to_numeric(cost_df['Fat(g)'], 'Fat(g)')))
vitamin_c = dict(zip(foods, to_numeric(cost_df['VitaminC(mg)'], 'VitaminC(mg)')))
cost = dict(zip(foods, to_numeric(cost_df['Cost'], 'Cost')))
m = gp.Model('MealPlanMinCost')
x_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i] * x_vars[i] for i in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[i] * x_vars[i] for i in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[i] * x_vars[i] for i in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[i] * x_vars[i] for i in foods)) >= 60, name='vitaminc_min')
m.addConstr(gp.quicksum((fat[i] * x_vars[i] for i in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: ${m.objVal:.2f}')
    print('--- Optimal Meal Plan (servings) ---')
    for i in foods:
        val = x_vars[i].X
        if val > 1e-05:
            print(f'{i}: {val:.3f} servings')
else:
    print(f'No optimal solution found. Status: {m.status}')