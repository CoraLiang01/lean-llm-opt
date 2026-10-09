import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv', sep=',')

def normalize_col(col):
    return re.sub('\\s+', '', col.strip().lower())
col_map = {normalize_col(col): col for col in cost_df.columns}
food_col = col_map.get('food')
calories_col = col_map.get('calories')
protein_col = col_map.get('proteing)')
fat_col = col_map.get('fatg)')
vitc_col = col_map.get('vitamincmg)')
cost_col = col_map.get('cost')
if not all([food_col, calories_col, protein_col, fat_col, vitc_col, cost_col]):
    raise KeyError('One or more required columns are missing in cost.csv.')
foods = cost_df[food_col].astype(str).tolist()
calories = cost_df.set_index(food_col)[calories_col].astype(float).to_dict()
protein = cost_df.set_index(food_col)[protein_col].astype(float).to_dict()
fat = cost_df.set_index(food_col)[fat_col].astype(float).to_dict()
vitc = cost_df.set_index(food_col)[vitc_col].astype(float).to_dict()
cost = cost_df.set_index(food_col)[cost_col].astype(float).to_dict()
for f in foods:
    for (param, d) in [('Calories', calories), ('Protein', protein), ('Fat', fat), ('VitaminC', vitc), ('Cost', cost)]:
        if f not in d or pd.isnull(d[f]):
            raise ValueError(f"Missing {param} data for food '{f}'.")
m = gp.Model('DietProblem')
x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * x[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * x[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[f] * x[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitc[f] * x[f] for f in foods)) >= 60, name='vitc_min')
m.addConstr(gp.quicksum((fat[f] * x[f] for f in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: ${m.objVal:.2f}')
    print('--- Optimal Meal Plan (servings per food) ---')
    for f in foods:
        if x[f].X > 1e-05:
            print(f'{f}: {x[f].X:.3f} servings')
    total_cal = sum((calories[f] * x[f].X for f in foods))
    total_prot = sum((protein[f] * x[f].X for f in foods))
    total_fat = sum((fat[f] * x[f].X for f in foods))
    total_vitc = sum((vitc[f] * x[f].X for f in foods))
    print('\n--- Achieved Nutrient Totals ---')
    print(f'Calories: {total_cal:.1f} kcal')
    print(f'Protein: {total_prot:.1f} g')
    print(f'Fat: {total_fat:.1f} g')
    print(f'Vitamin C: {total_vitc:.1f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')