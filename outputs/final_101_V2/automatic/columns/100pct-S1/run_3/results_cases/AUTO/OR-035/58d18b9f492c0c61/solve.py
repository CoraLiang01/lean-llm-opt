import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S1/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',')
foods = df['Food'].astype(str).tolist()

def get_col(colname):
    for c in df.columns:
        if re.sub('[\\s\\(\\)]', '', c).casefold() == re.sub('[\\s\\(\\)]', '', colname).casefold():
            return c
    raise KeyError(f"Column '{colname}' not found in CSV.")
calories_col = get_col('Calories')
protein_col = get_col('Protein(g)')
fat_col = get_col('Fat(g)')
vitc_col = get_col('VitaminC(mg)')
cost_col = get_col('Cost')
calories = dict(zip(df['Food'].astype(str), df[calories_col].astype(float)))
protein = dict(zip(df['Food'].astype(str), df[protein_col].astype(float)))
fat = dict(zip(df['Food'].astype(str), df[fat_col].astype(float)))
vitc = dict(zip(df['Food'].astype(str), df[vitc_col].astype(float)))
cost = dict(zip(df['Food'].astype(str), df[cost_col].astype(float)))
for f in foods:
    for d, name in [(calories, 'Calories'), (protein, 'Protein(g)'), (fat, 'Fat(g)'), (vitc, 'VitaminC(mg)'), (cost, 'Cost')]:
        if f not in d:
            raise ValueError(f"Missing {name} data for food '{f}'.")
m = gp.Model('DietProblem')
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
            print(f'{f}: {val:.4f} servings')
    total_cal = sum((calories[f] * x[f].X for f in foods))
    total_prot = sum((protein[f] * x[f].X for f in foods))
    total_fat = sum((fat[f] * x[f].X for f in foods))
    total_vitc = sum((vitc[f] * x[f].X for f in foods))
    print('\n--- Achieved Nutrition Totals ---')
    print(f'Calories:   {total_cal:.2f} kcal')
    print(f'Protein:    {total_prot:.2f} g')
    print(f'Fat:        {total_fat:.2f} g')
    print(f'Vitamin C:  {total_vitc:.2f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')