import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',')
if df['Food'].isnull().any():
    raise ValueError('Missing Food identifiers in cost.csv')
foods = df['Food'].astype(str).tolist()
n_foods = len(foods)

def get_col(colname, required=True):
    for c in df.columns:
        if c.strip().casefold() == colname.strip().casefold():
            return c
    if required:
        raise KeyError(f"Required column '{colname}' not found in cost.csv")
    return None
calories_col = get_col('Calories')
protein_col = get_col('Protein(g)')
fat_col = get_col('Fat(g)')
vitc_col = get_col('VitaminC(mg)')
cost_col = get_col('Cost')
calories = df.set_index('Food')[calories_col].astype(float).to_dict()
protein = df.set_index('Food')[protein_col].astype(float).to_dict()
fat = df.set_index('Food')[fat_col].astype(float).to_dict()
vitc = df.set_index('Food')[vitc_col].astype(float).to_dict()
cost = df.set_index('Food')[cost_col].astype(float).to_dict()
for f in foods:
    for (pname, pdict) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitc), ('Cost', cost)]:
        if f not in pdict or pd.isnull(pdict[f]):
            raise ValueError(f"Missing or NaN value for '{pname}' in food '{f}'")
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
        servings = x[f].X
        if servings > 1e-05:
            print(f'{f}: {servings:.4f} servings')
    total_cal = sum((calories[f] * x[f].X for f in foods))
    total_prot = sum((protein[f] * x[f].X for f in foods))
    total_vitc = sum((vitc[f] * x[f].X for f in foods))
    total_fat = sum((fat[f] * x[f].X for f in foods))
    print('\n--- Achieved Nutrient Totals ---')
    print(f'Calories:   {total_cal:.2f} kcal')
    print(f'Protein:    {total_prot:.2f} g')
    print(f'Vitamin C:  {total_vitc:.2f} mg')
    print(f'Fat:        {total_fat:.2f} g')
else:
    print(f'No optimal solution found. Status: {m.status}')