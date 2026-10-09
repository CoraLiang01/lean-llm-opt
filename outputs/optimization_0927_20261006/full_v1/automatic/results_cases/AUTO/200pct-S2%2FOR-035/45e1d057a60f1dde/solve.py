import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
food_ids = df['Food'].tolist()

def to_float_col(df, col):
    return df[col].apply(lambda x: float(str(x).strip()))
calories = dict(zip(food_ids, to_float_col(df, 'Calories')))
protein = dict(zip(food_ids, to_float_col(df, 'Protein(g)')))
fat = dict(zip(food_ids, to_float_col(df, 'Fat(g)')))
vitamin_c = dict(zip(food_ids, to_float_col(df, 'VitaminC(mg)')))
cost = dict(zip(food_ids, to_float_col(df, 'Cost')))
for i in food_ids:
    for (param, d) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
        if i not in d:
            raise ValueError(f"Missing {param} for food '{i}'")
        if not np.isfinite(d[i]):
            raise ValueError(f"Non-finite {param} for food '{i}'")
m = gp.Model('MealPlanMinCost')
servings_vars = m.addVars(food_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i] * servings_vars[i] for i in food_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[i] * servings_vars[i] for i in food_ids)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[i] * servings_vars[i] for i in food_ids)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[i] * servings_vars[i] for i in food_ids)) >= 60, name='vitaminc_min')
m.addConstr(gp.quicksum((fat[i] * servings_vars[i] for i in food_ids)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for i in food_ids:
        val = servings_vars[i].X
        if val > 1e-05:
            print(f'{i}: {val:.4f} servings')
    total_cal = sum((calories[i] * servings_vars[i].X for i in food_ids))
    total_prot = sum((protein[i] * servings_vars[i].X for i in food_ids))
    total_fat = sum((fat[i] * servings_vars[i].X for i in food_ids))
    total_vitc = sum((vitamin_c[i] * servings_vars[i].X for i in food_ids))
    print('\n--- Achieved Nutrition Totals ---')
    print(f'Calories: {total_cal:.2f} kcal')
    print(f'Protein: {total_prot:.2f} g')
    print(f'Fat: {total_fat:.2f} g')
    print(f'Vitamin C: {total_vitc:.2f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')