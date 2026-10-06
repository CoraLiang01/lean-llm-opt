import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S2/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',')
foods = df['Food'].astype(str).tolist()

def get_col(colname, dfcols):
    pat = re.compile('\\b' + re.escape(colname).replace('\\ ', '\\s*').replace('\\(', '\\s*\\(?').replace('\\)', '\\)?') + '\\b', re.IGNORECASE)
    for c in dfcols:
        if pat.search(c.replace(' ', '').replace('(', '').replace(')', '')):
            return c
    raise KeyError(f"Column '{colname}' not found in CSV columns: {dfcols}")
col_calories = get_col('Calories', df.columns)
col_protein = get_col('Protein(g)', df.columns)
col_fat = get_col('Fat(g)', df.columns)
col_vitc = get_col('VitaminC(mg)', df.columns)
col_cost = get_col('Cost', df.columns)
calories = df.set_index('Food')[col_calories].astype(float).to_dict()
protein = df.set_index('Food')[col_protein].astype(float).to_dict()
fat = df.set_index('Food')[col_fat].astype(float).to_dict()
vitc = df.set_index('Food')[col_vitc].astype(float).to_dict()
cost = df.set_index('Food')[col_cost].astype(float).to_dict()
for f in foods:
    for d, name in [(calories, 'Calories'), (protein, 'Protein(g)'), (fat, 'Fat(g)'), (vitc, 'VitaminC(mg)'), (cost, 'Cost')]:
        if f not in d:
            raise ValueError(f"Missing {name} data for food '{f}'")
m = gp.Model('DietPlan')
x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * x[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * x[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[f] * x[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitc[f] * x[f] for f in foods)) >= 60, name='vitc_min')
m.addConstr(gp.quicksum((fat[f] * x[f] for f in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for f in foods:
        val = x[f].X
        if val > 1e-05:
            print(f'{f}: {val:.3f} servings')
    total_cal = sum((calories[f] * x[f].X for f in foods))
    total_prot = sum((protein[f] * x[f].X for f in foods))
    total_fat = sum((fat[f] * x[f].X for f in foods))
    total_vitc = sum((vitc[f] * x[f].X for f in foods))
    print('\n--- Nutrient Totals ---')
    print(f'Calories:   {total_cal:.1f} kcal')
    print(f'Protein:    {total_prot:.1f} g')
    print(f'Fat:        {total_fat:.1f} g')
    print(f'Vitamin C:  {total_vitc:.1f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')