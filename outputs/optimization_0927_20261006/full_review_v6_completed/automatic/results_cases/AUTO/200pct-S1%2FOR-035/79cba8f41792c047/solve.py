import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
if 'Food' not in df.columns:
    raise KeyError("Required column 'Food' not found in cost.csv")
foods = df['Food'].tolist()

def get_numeric_series(df, colname, foods):
    if colname not in df.columns:
        raise KeyError(f"Required column '{colname}' not found in cost.csv")
    vals = pd.to_numeric(df[colname], errors='raise')
    return dict(zip(df['Food'], vals))
calories = get_numeric_series(df, 'Calories', foods)
protein = get_numeric_series(df, 'Protein(g)', foods)
fat = get_numeric_series(df, 'Fat(g)', foods)
vitamin_c = get_numeric_series(df, 'VitaminC(mg)', foods)
cost = get_numeric_series(df, 'Cost', foods)
for food in foods:
    for (param, param_dict) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
        if food not in param_dict:
            raise ValueError(f"Missing {param} value for food '{food}'")
m = gp.Model('MealPlanMinCost')
servings_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[food] * servings_vars[food] for food in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[food] * servings_vars[food] for food in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[food] * servings_vars[food] for food in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[food] * servings_vars[food] for food in foods)) >= 60, name='vitamin_c_min')
m.addConstr(gp.quicksum((fat[food] * servings_vars[food] for food in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for food in foods:
        val = servings_vars[food].X
        if val > 1e-05:
            print(f'{food}: {val:.3f} servings')
    total_cal = sum((calories[food] * servings_vars[food].X for food in foods))
    total_prot = sum((protein[food] * servings_vars[food].X for food in foods))
    total_fat = sum((fat[food] * servings_vars[food].X for food in foods))
    total_vitc = sum((vitamin_c[food] * servings_vars[food].X for food in foods))
    print('\n--- Achieved Nutrition Totals ---')
    print(f'Calories: {total_cal:.1f} kcal')
    print(f'Protein: {total_prot:.1f} g')
    print(f'Fat: {total_fat:.1f} g')
    print(f'Vitamin C: {total_vitc:.1f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')