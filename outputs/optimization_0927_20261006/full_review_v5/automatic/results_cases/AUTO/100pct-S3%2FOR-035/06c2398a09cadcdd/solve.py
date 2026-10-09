import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
foods = df['Food'].tolist()

def get_numeric_param(colname):
    if colname not in df.columns:
        raise KeyError(f"Required column '{colname}' not found in cost.csv")
    return {row['Food']: float(row[colname]) for (_, row) in df.iterrows()}
cost_param = get_numeric_param('Cost')
calories_param = get_numeric_param('Calories')
protein_param = get_numeric_param('Protein(g)')
fat_param = get_numeric_param('Fat(g)')
vitc_param = get_numeric_param('VitaminC(mg)')
for food in foods:
    for (pname, param) in [('Cost', cost_param), ('Calories', calories_param), ('Protein(g)', protein_param), ('Fat(g)', fat_param), ('VitaminC(mg)', vitc_param)]:
        if food not in param:
            raise ValueError(f"Missing {pname} for food '{food}'")
m = gp.Model('OneDayMealPlan')
servings_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost_param[f] * servings_vars[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories_param[f] * servings_vars[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein_param[f] * servings_vars[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitc_param[f] * servings_vars[f] for f in foods)) >= 60, name='vitc_min')
m.addConstr(gp.quicksum((fat_param[f] * servings_vars[f] for f in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: ${m.objVal:.2f}')
    print('--- Meal Plan (servings per food) ---')
    for f in foods:
        val = servings_vars[f].X
        if val > 1e-05:
            print(f'{f}: {val:.4f} servings')
    total_cal = sum((calories_param[f] * servings_vars[f].X for f in foods))
    total_prot = sum((protein_param[f] * servings_vars[f].X for f in foods))
    total_fat = sum((fat_param[f] * servings_vars[f].X for f in foods))
    total_vitc = sum((vitc_param[f] * servings_vars[f].X for f in foods))
    print('\n--- Achieved Nutrient Totals ---')
    print(f'Calories: {total_cal:.2f} kcal')
    print(f'Protein: {total_prot:.2f} g')
    print(f'Fat: {total_fat:.2f} g')
    print(f'Vitamin C: {total_vitc:.2f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')