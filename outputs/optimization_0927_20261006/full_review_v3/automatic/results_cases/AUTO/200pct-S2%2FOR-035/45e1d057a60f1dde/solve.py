import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['Food', 'Calories', 'Protein(g)', 'Fat(g)', 'VitaminC(mg)', 'Cost']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
df['Food'] = df['Food'].astype(str)
df = df.set_index('Food', drop=False)
foods = list(df.index)

def to_numeric_checked(series, colname):
    try:
        return pd.to_numeric(series, errors='raise')
    except Exception as e:
        raise ValueError(f"Column '{colname}' contains non-numeric values: {e}")
calories = to_numeric_checked(df['Calories'], 'Calories').to_dict()
protein = to_numeric_checked(df['Protein(g)'], 'Protein(g)').to_dict()
fat = to_numeric_checked(df['Fat(g)'], 'Fat(g)').to_dict()
vitamin_c = to_numeric_checked(df['VitaminC(mg)'], 'VitaminC(mg)').to_dict()
cost = to_numeric_checked(df['Cost'], 'Cost').to_dict()
for f in foods:
    for (param, param_dict) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
        if f not in param_dict:
            raise KeyError(f"Food '{f}' missing parameter '{param}'.")
m = gp.Model('OneDayMealPlan')
x_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * x_vars[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * x_vars[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[f] * x_vars[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[f] * x_vars[f] for f in foods)) >= 60, name='vitaminc_min')
m.addConstr(gp.quicksum((fat[f] * x_vars[f] for f in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: ${m.objVal:.2f}')
    print('--- Meal Plan (servings per food) ---')
    for f in foods:
        servings = x_vars[f].X
        if servings > 1e-05:
            print(f'{f}: {servings:.3f} servings')
    total_cal = sum((calories[f] * x_vars[f].X for f in foods))
    total_prot = sum((protein[f] * x_vars[f].X for f in foods))
    total_fat = sum((fat[f] * x_vars[f].X for f in foods))
    total_vitc = sum((vitamin_c[f] * x_vars[f].X for f in foods))
    print('\n--- Nutrient Totals ---')
    print(f'Calories: {total_cal:.1f} kcal')
    print(f'Protein: {total_prot:.1f} g')
    print(f'Fat: {total_fat:.1f} g')
    print(f'Vitamin C: {total_vitc:.1f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')