import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S1/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',')
foods = df['Food'].astype(str).tolist()

def get_col(df, colname):
    for c in df.columns:
        if re.sub('\\s+|\\(.*?\\)', '', c).casefold() == re.sub('\\s+|\\(.*?\\)', '', colname).casefold():
            return c
    raise KeyError(f"Column '{colname}' not found in CSV.")
calories_col = get_col(df, 'Calories')
protein_col = get_col(df, 'Protein(g)')
fat_col = get_col(df, 'Fat(g)')
vitc_col = get_col(df, 'VitaminC(mg)')
cost_col = get_col(df, 'Cost')
for col in [calories_col, protein_col, fat_col, vitc_col, cost_col]:
    if df[col].isnull().any():
        raise ValueError(f"Missing values found in column '{col}'.")
Calories = df.set_index('Food')[calories_col].astype(float).to_dict()
Protein = df.set_index('Food')[protein_col].astype(float).to_dict()
Fat = df.set_index('Food')[fat_col].astype(float).to_dict()
VitaminC = df.set_index('Food')[vitc_col].astype(float).to_dict()
Cost = df.set_index('Food')[cost_col].astype(float).to_dict()
for food in foods:
    for param, d in [('Calories', Calories), ('Protein', Protein), ('Fat', Fat), ('VitaminC', VitaminC), ('Cost', Cost)]:
        if food not in d:
            raise KeyError(f"Food '{food}' missing parameter '{param}'.")
m = gp.Model('DietProblem')
x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((Cost[i] * x[i] for i in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((Calories[i] * x[i] for i in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((Protein[i] * x[i] for i in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((VitaminC[i] * x[i] for i in foods)) >= 60, name='vitc_min')
m.addConstr(gp.quicksum((Fat[i] * x[i] for i in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for i in foods:
        if x[i].X > 1e-05:
            print(f'{i}: {x[i].X:.4f} servings')
    total_cal = sum((Calories[i] * x[i].X for i in foods))
    total_prot = sum((Protein[i] * x[i].X for i in foods))
    total_fat = sum((Fat[i] * x[i].X for i in foods))
    total_vitc = sum((VitaminC[i] * x[i].X for i in foods))
    print('\n--- Achieved Nutrition Totals ---')
    print(f'Calories: {total_cal:.2f} kcal')
    print(f'Protein: {total_prot:.2f} g')
    print(f'Fat: {total_fat:.2f} g')
    print(f'Vitamin C: {total_vitc:.2f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')