import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
foods = df['Food'].tolist()

def to_float_series(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Column '{colname}' could not be fully converted to float: {e}")
calories = to_float_series(df['Calories'], 'Calories')
protein = to_float_series(df['Protein(g)'], 'Protein(g)')
fat = to_float_series(df['Fat(g)'], 'Fat(g)')
vitc = to_float_series(df['VitaminC(mg)'], 'VitaminC(mg)')
cost = to_float_series(df['Cost'], 'Cost')
calories_dict = dict(zip(foods, calories))
protein_dict = dict(zip(foods, protein))
fat_dict = dict(zip(foods, fat))
vitc_dict = dict(zip(foods, vitc))
cost_dict = dict(zip(foods, cost))
m = gp.Model('OneDayMealPlan')
servings_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost_dict[f] * servings_vars[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories_dict[f] * servings_vars[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein_dict[f] * servings_vars[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitc_dict[f] * servings_vars[f] for f in foods)) >= 60, name='vitc_min')
m.addConstr(gp.quicksum((fat_dict[f] * servings_vars[f] for f in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for f in foods:
        val = servings_vars[f].X
        if val > 1e-05:
            print(f'{f}: {val:.4f} servings')
    total_cal = sum((calories_dict[f] * servings_vars[f].X for f in foods))
    total_prot = sum((protein_dict[f] * servings_vars[f].X for f in foods))
    total_fat = sum((fat_dict[f] * servings_vars[f].X for f in foods))
    total_vitc = sum((vitc_dict[f] * servings_vars[f].X for f in foods))
    print('\n--- Achieved Nutrition Totals ---')
    print(f'Calories: {total_cal:.2f} kcal')
    print(f'Protein: {total_prot:.2f} g')
    print(f'Fat: {total_fat:.2f} g')
    print(f'Vitamin C: {total_vitc:.2f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')