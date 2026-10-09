import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)

def norm_col(col):
    return col.strip().casefold()
col_map = {}
for col in df.columns:
    ncol = norm_col(col)
    if ncol == 'food':
        col_map['food'] = col
    elif ncol == 'calories':
        col_map['calories'] = col
    elif ncol == 'protein(g)':
        col_map['protein'] = col
    elif ncol == 'fat(g)':
        col_map['fat'] = col
    elif ncol == 'vitaminc(mg)':
        col_map['vitaminc'] = col
    elif ncol == 'cost':
        col_map['cost'] = col
required_cols = ['food', 'calories', 'protein', 'fat', 'vitaminc', 'cost']
missing = [k for k in required_cols if k not in col_map]
if missing:
    raise KeyError(f'Missing required columns in CSV: {missing}')
foods = df[col_map['food']].tolist()

def to_float_series(series, colname):
    try:
        return pd.to_numeric(series, errors='raise')
    except Exception as e:
        raise ValueError(f"Column '{colname}' contains non-numeric values.") from e
calories = dict(zip(foods, to_float_series(df[col_map['calories']], col_map['calories'])))
protein = dict(zip(foods, to_float_series(df[col_map['protein']], col_map['protein'])))
fat = dict(zip(foods, to_float_series(df[col_map['fat']], col_map['fat'])))
vitaminc = dict(zip(foods, to_float_series(df[col_map['vitaminc']], col_map['vitaminc'])))
cost = dict(zip(foods, to_float_series(df[col_map['cost']], col_map['cost'])))
for f in foods:
    for (pname, pdict) in [('calories', calories), ('protein', protein), ('fat', fat), ('vitaminc', vitaminc), ('cost', cost)]:
        if f not in pdict:
            raise KeyError(f"Missing {pname} data for food '{f}'.")
m = gp.Model('MealPlanMinCost')
x_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * x_vars[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * x_vars[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[f] * x_vars[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitaminc[f] * x_vars[f] for f in foods)) >= 60, name='vitaminc_min')
m.addConstr(gp.quicksum((fat[f] * x_vars[f] for f in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.4f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for f in foods:
        val = x_vars[f].X
        if val > 1e-05:
            print(f'{f}: {val:.4f} servings')
    total_cal = sum((calories[f] * x_vars[f].X for f in foods))
    total_prot = sum((protein[f] * x_vars[f].X for f in foods))
    total_fat = sum((fat[f] * x_vars[f].X for f in foods))
    total_vitc = sum((vitaminc[f] * x_vars[f].X for f in foods))
    print('\n--- Achieved Nutrient Totals ---')
    print(f'Calories: {total_cal:.2f} kcal')
    print(f'Protein: {total_prot:.2f} g')
    print(f'Fat: {total_fat:.2f} g')
    print(f'Vitamin C: {total_vitc:.2f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')