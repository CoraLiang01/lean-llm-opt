import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv', dtype=str, keep_default_na=False)
foods = cost_df['Food'].tolist()

def to_float_col(df, col):
    try:
        return df[col].astype(float)
    except Exception as e:
        raise ValueError(f"Column '{col}' could not be converted to float: {e}")
calories = dict(zip(foods, to_float_col(cost_df, 'Calories')))
protein = dict(zip(foods, to_float_col(cost_df, 'Protein(g)')))
fat = dict(zip(foods, to_float_col(cost_df, 'Fat(g)')))
vitamin_c = dict(zip(foods, to_float_col(cost_df, 'VitaminC(mg)')))
cost = dict(zip(foods, to_float_col(cost_df, 'Cost')))
m = gp.Model('MealPlan')
x_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * x_vars[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * x_vars[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[f] * x_vars[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[f] * x_vars[f] for f in foods)) >= 60, name='vitaminC_min')
m.addConstr(gp.quicksum((fat[f] * x_vars[f] for f in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for f in foods:
        val = x_vars[f].X
        if val > 1e-06:
            print(f'  {f}: {val:.4f} servings')
else:
    print(f'No optimal solution found. Status: {m.status}')