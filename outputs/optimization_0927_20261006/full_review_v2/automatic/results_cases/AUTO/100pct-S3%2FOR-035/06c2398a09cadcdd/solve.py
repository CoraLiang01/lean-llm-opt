import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Others_example/Others4/cost.csv'
cost_df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)

def norm_col(col):
    return re.sub('\\s+', '', col.strip().casefold())
col_map = {norm_col(col): col for col in cost_df.columns}
required_cols = {'food': None, 'calories': None, 'protein(g)': None, 'fat(g)': None, 'vitaminc(mg)': None, 'cost': None}
for key in required_cols:
    found = [col for col in cost_df.columns if norm_col(col) == key]
    if not found:
        raise KeyError(f"Required column '{key}' not found in CSV columns: {list(cost_df.columns)}")
    required_cols[key] = found[0]
foods = cost_df[required_cols['food']].tolist()

def to_float_series(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Column '{colname}' could not be fully converted to float: {e}")
calories = dict(zip(foods, to_float_series(cost_df[required_cols['calories']], required_cols['calories'])))
protein = dict(zip(foods, to_float_series(cost_df[required_cols['protein(g)']], required_cols['protein(g)'])))
fat = dict(zip(foods, to_float_series(cost_df[required_cols['fat(g)']], required_cols['fat(g)'])))
vitaminc = dict(zip(foods, to_float_series(cost_df[required_cols['vitaminc(mg)']], required_cols['vitaminc(mg)'])))
cost = dict(zip(foods, to_float_series(cost_df[required_cols['cost']], required_cols['cost'])))
for f in foods:
    for (pname, pdict) in [('Calories', calories), ('Protein', protein), ('Fat', fat), ('VitaminC', vitaminc), ('Cost', cost)]:
        if f not in pdict:
            raise KeyError(f"Food '{f}' missing parameter '{pname}'.")
m = gp.Model('MealPlanMinCost')
x_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * x_vars[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * x_vars[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[f] * x_vars[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitaminc[f] * x_vars[f] for f in foods)) >= 60, name='vitaminc_min')
m.addConstr(gp.quicksum((fat[f] * x_vars[f] for f in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for f in foods:
        val = x_vars[f].X
        if val > 1e-05:
            print(f'  {f}: {val:.4f} servings')
    total_cal = sum((calories[f] * x_vars[f].X for f in foods))
    total_prot = sum((protein[f] * x_vars[f].X for f in foods))
    total_fat = sum((fat[f] * x_vars[f].X for f in foods))
    total_vitc = sum((vitaminc[f] * x_vars[f].X for f in foods))
    print('\n--- Achieved Nutrition Totals ---')
    print(f'  Calories:   {total_cal:.2f} kcal')
    print(f'  Protein:    {total_prot:.2f} g')
    print(f'  Fat:        {total_fat:.2f} g')
    print(f'  Vitamin C:  {total_vitc:.2f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')