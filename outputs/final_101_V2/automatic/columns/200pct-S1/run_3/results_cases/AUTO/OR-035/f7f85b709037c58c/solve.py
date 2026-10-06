import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S1/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',')
required_cols = {'Food': 'Food', 'Calories': 'Calories', 'Protein(g)': 'Protein(g)', 'Fat(g)': 'Fat(g)', 'VitaminC(mg)': 'VitaminC(mg)', 'Cost': 'Cost'}
for col in required_cols.values():
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
foods = df['Food'].astype(str).tolist()
calories = df.set_index('Food')['Calories'].astype(float).to_dict()
protein = df.set_index('Food')['Protein(g)'].astype(float).to_dict()
fat = df.set_index('Food')['Fat(g)'].astype(float).to_dict()
vitaminc = df.set_index('Food')['VitaminC(mg)'].astype(float).to_dict()
cost = df.set_index('Food')['Cost'].astype(float).to_dict()
for f in foods:
    for param, d in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitaminc), ('Cost', cost)]:
        if f not in d or pd.isnull(d[f]):
            raise ValueError(f"Missing value for {param} in food '{f}'.")
m = gp.Model('OneDayMealPlan')
x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * x[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * x[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[f] * x[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitaminc[f] * x[f] for f in foods)) >= 60, name='vitaminc_min')
m.addConstr(gp.quicksum((fat[f] * x[f] for f in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for f in foods:
        servings = x[f].X
        if servings > 1e-05:
            print(f'{f}: {servings:.3f} servings')
    total_cal = sum((calories[f] * x[f].X for f in foods))
    total_prot = sum((protein[f] * x[f].X for f in foods))
    total_fat = sum((fat[f] * x[f].X for f in foods))
    total_vitc = sum((vitaminc[f] * x[f].X for f in foods))
    print('\n--- Achieved Nutrition Totals ---')
    print(f'Calories:   {total_cal:.1f} kcal')
    print(f'Protein:    {total_prot:.1f} g')
    print(f'Fat:        {total_fat:.1f} g')
    print(f'Vitamin C:  {total_vitc:.1f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')