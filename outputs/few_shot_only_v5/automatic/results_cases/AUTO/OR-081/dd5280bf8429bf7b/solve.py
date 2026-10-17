import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
food_ids = df['Food'].tolist()

def to_float_col(df, col):
    try:
        return df[col].astype(float).to_dict()
    except Exception as e:
        raise ValueError(f"Column '{col}' could not be converted to float: {e}")
calories = to_float_col(df.set_index('Food'), 'Calories')
protein = to_float_col(df.set_index('Food'), 'Protein(g)')
fat = to_float_col(df.set_index('Food'), 'Fat(g)')
vitaminc = to_float_col(df.set_index('Food'), 'VitaminC(mg)')
cost = to_float_col(df.set_index('Food'), 'Cost')
for i in food_ids:
    for param, param_dict in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitaminc), ('Cost', cost)]:
        if i not in param_dict:
            raise KeyError(f"Food '{i}' missing parameter '{param}'.")
m = gp.Model('MealPlanMinCost')
x_vars = m.addVars(food_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i] * x_vars[i] for i in food_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[i] * x_vars[i] for i in food_ids)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[i] * x_vars[i] for i in food_ids)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitaminc[i] * x_vars[i] for i in food_ids)) >= 60, name='vitaminc_min')
m.addConstr(gp.quicksum((fat[i] * x_vars[i] for i in food_ids)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: ${m.objVal:.2f}')
    print('--- Optimal Meal Plan (servings per food) ---')
    for i in food_ids:
        val = x_vars[i].X
        if val > 1e-05:
            print(f'{i}: {val:.4f} servings')
    total_cal = sum((calories[i] * x_vars[i].X for i in food_ids))
    total_prot = sum((protein[i] * x_vars[i].X for i in food_ids))
    total_fat = sum((fat[i] * x_vars[i].X for i in food_ids))
    total_vitc = sum((vitaminc[i] * x_vars[i].X for i in food_ids))
    print('\n--- Achieved Nutrition Totals ---')
    print(f'Calories: {total_cal:.2f} kcal')
    print(f'Protein: {total_prot:.2f} g')
    print(f'Fat: {total_fat:.2f} g')
    print(f'Vitamin C: {total_vitc:.2f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')