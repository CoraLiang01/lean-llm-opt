import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv', sep=',')
foods = cost_df['Food'].astype(str).tolist()
calories = dict(zip(cost_df['Food'].astype(str), cost_df['Calories'].astype(float)))
protein = dict(zip(cost_df['Food'].astype(str), cost_df['Protein(g)'].astype(float)))
fat = dict(zip(cost_df['Food'].astype(str), cost_df['Fat(g)'].astype(float)))
vitamin_c = dict(zip(cost_df['Food'].astype(str), cost_df['VitaminC(mg)'].astype(float)))
cost = dict(zip(cost_df['Food'].astype(str), cost_df['Cost'].astype(float)))
for f in foods:
    for param, d in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
        if f not in d or pd.isnull(d[f]):
            raise ValueError(f"Missing value for {param} in food '{f}'.")
m = gp.Model('MealPlanMinCost')
x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * x[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * x[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[f] * x[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[f] * x[f] for f in foods)) >= 60, name='vitaminc_min')
m.addConstr(gp.quicksum((fat[f] * x[f] for f in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: ${m.objVal:.2f}')
    print('\n--- Meal Plan (servings per food) ---')
    for f in foods:
        servings = x[f].X
        if servings > 1e-05:
            print(f'{f}: {servings:.3f} servings')
    total_cal = sum((calories[f] * x[f].X for f in foods))
    total_prot = sum((protein[f] * x[f].X for f in foods))
    total_fat = sum((fat[f] * x[f].X for f in foods))
    total_vitc = sum((vitamin_c[f] * x[f].X for f in foods))
    print('\n--- Achieved Nutrient Totals ---')
    print(f'Calories: {total_cal:.1f} kcal (min 2000)')
    print(f'Protein: {total_prot:.1f} g (min 50)')
    print(f'Fat: {total_fat:.1f} g (max 70)')
    print(f'Vitamin C: {total_vitc:.1f} mg (min 60)')
else:
    print(f'No optimal solution found. Status: {m.status}')