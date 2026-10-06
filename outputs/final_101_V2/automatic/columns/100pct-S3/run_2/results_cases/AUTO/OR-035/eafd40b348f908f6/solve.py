import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S3/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',')
foods = df['Food'].astype(str).tolist()

def get_param_dict(df, colname):
    return dict(zip(df['Food'].astype(str), df[colname]))
cost = get_param_dict(df, 'Cost')
calories = get_param_dict(df, 'Calories')
protein = get_param_dict(df, 'Protein(g)')
fat = get_param_dict(df, 'Fat(g)')
vitamin_c = get_param_dict(df, 'VitaminC(mg)')
for param_name, param_dict in [('Cost', cost), ('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c)]:
    if set(param_dict.keys()) != set(foods):
        raise ValueError(f"Parameter '{param_name}' missing data for some foods.")
m = gp.Model('DietPlan')
x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i] * x[i] for i in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[i] * x[i] for i in foods)) >= 2000, name='cal_min')
m.addConstr(gp.quicksum((protein[i] * x[i] for i in foods)) >= 50, name='prot_min')
m.addConstr(gp.quicksum((vitamin_c[i] * x[i] for i in foods)) >= 60, name='vitc_min')
m.addConstr(gp.quicksum((fat[i] * x[i] for i in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for i in foods:
        if x[i].X > 1e-05:
            print(f'{i}: {x[i].X:.4f} servings')
    total_cal = sum((calories[i] * x[i].X for i in foods))
    total_prot = sum((protein[i] * x[i].X for i in foods))
    total_fat = sum((fat[i] * x[i].X for i in foods))
    total_vitc = sum((vitamin_c[i] * x[i].X for i in foods))
    print('\n--- Nutrient Totals ---')
    print(f'Calories:   {total_cal:.2f} kcal')
    print(f'Protein:    {total_prot:.2f} g')
    print(f'Fat:        {total_fat:.2f} g')
    print(f'Vitamin C:  {total_vitc:.2f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')