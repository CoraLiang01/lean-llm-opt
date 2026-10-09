import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv'
cost_df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
foods = cost_df['Food'].tolist()

def to_float_series(df, col):
    return df[col].astype(str).str.strip().replace('', np.nan).astype(float)
calories = to_float_series(cost_df, 'Calories')
protein = to_float_series(cost_df, 'Protein(g)')
fat = to_float_series(cost_df, 'Fat(g)')
vitamin_c = to_float_series(cost_df, 'VitaminC(mg)')
cost = to_float_series(cost_df, 'Cost')
calories_dict = dict(zip(foods, calories))
protein_dict = dict(zip(foods, protein))
fat_dict = dict(zip(foods, fat))
vitamin_c_dict = dict(zip(foods, vitamin_c))
cost_dict = dict(zip(foods, cost))
m = gp.Model('MealPlanMinCost')
x_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost_dict[i] * x_vars[i] for i in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories_dict[i] * x_vars[i] for i in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein_dict[i] * x_vars[i] for i in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c_dict[i] * x_vars[i] for i in foods)) >= 60, name='vitaminc_min')
m.addConstr(gp.quicksum((fat_dict[i] * x_vars[i] for i in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: ${m.objVal:.2f}')
    print('\n--- Optimal Meal Plan (Servings) ---')
    for i in foods:
        val = x_vars[i].X
        if val > 1e-05:
            print(f'  {i}: {val:.3f} servings')
    total_cal = sum((calories_dict[i] * x_vars[i].X for i in foods))
    total_prot = sum((protein_dict[i] * x_vars[i].X for i in foods))
    total_fat = sum((fat_dict[i] * x_vars[i].X for i in foods))
    total_vitc = sum((vitamin_c_dict[i] * x_vars[i].X for i in foods))
    print('\n--- Achieved Nutrition Totals ---')
    print(f'  Calories:   {total_cal:.1f} kcal')
    print(f'  Protein:    {total_prot:.1f} g')
    print(f'  Fat:        {total_fat:.1f} g')
    print(f'  Vitamin C:  {total_vitc:.1f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')