import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv'
cost_df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
if 'Food' not in cost_df.columns:
    raise KeyError("Required column 'Food' not found in cost.csv")
foods = cost_df['Food'].tolist()

def get_numeric_series(df, colname, foods):
    if colname not in df.columns:
        raise KeyError(f"Required column '{colname}' not found in cost.csv")
    vals = {}
    for idx, row in df.iterrows():
        food = row['Food']
        try:
            val = float(row[colname])
        except Exception as e:
            raise ValueError(f"Non-numeric value in column '{colname}' for food '{food}': {row[colname]}")
        vals[food] = val
    missing = set(foods) - set(vals.keys())
    if missing:
        raise KeyError(f'Missing parameter values for foods: {missing}')
    return vals
cost_param = get_numeric_series(cost_df, 'Cost', foods)
calories_param = get_numeric_series(cost_df, 'Calories', foods)
protein_param = get_numeric_series(cost_df, 'Protein(g)', foods)
fat_param = get_numeric_series(cost_df, 'Fat(g)', foods)
vitc_param = get_numeric_series(cost_df, 'VitaminC(mg)', foods)
m = gp.Model('MealPlanMinCost')
x_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost_param[f] * x_vars[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories_param[f] * x_vars[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein_param[f] * x_vars[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitc_param[f] * x_vars[f] for f in foods)) >= 60, name='vitc_min')
m.addConstr(gp.quicksum((fat_param[f] * x_vars[f] for f in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: ${m.objVal:.2f}')
    print('--- Optimal Meal Plan (servings) ---')
    for f in foods:
        val = x_vars[f].X
        if val > 1e-06:
            print(f'  {f}: {val:.3f} servings')
    total_cal = sum((calories_param[f] * x_vars[f].X for f in foods))
    total_prot = sum((protein_param[f] * x_vars[f].X for f in foods))
    total_fat = sum((fat_param[f] * x_vars[f].X for f in foods))
    total_vitc = sum((vitc_param[f] * x_vars[f].X for f in foods))
    print('\n--- Achieved Nutrient Totals ---')
    print(f'  Calories: {total_cal:.1f} kcal')
    print(f'  Protein: {total_prot:.1f} g')
    print(f'  Fat: {total_fat:.1f} g')
    print(f'  Vitamin C: {total_vitc:.1f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')