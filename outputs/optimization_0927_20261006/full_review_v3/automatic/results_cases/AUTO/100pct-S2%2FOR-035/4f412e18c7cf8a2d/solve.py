import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)

def norm_col(col):
    return col.strip().casefold()
col_map = {norm_col(c): c for c in df.columns}
required_cols = {'food': 'food', 'calories': 'calories', 'protein(g)': 'protein(g)', 'fat(g)': 'fat(g)', 'vitaminc(mg)': 'vitaminc(mg)', 'cost': 'cost'}
for (req, norm) in required_cols.items():
    if norm not in col_map:
        raise KeyError(f"Required column '{req}' not found in CSV.")
foods = df[col_map['food']].tolist()

def to_float_series(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Column '{colname}' could not be converted to float: {e}")
calories = dict(zip(foods, to_float_series(df[col_map['calories']], 'Calories')))
protein = dict(zip(foods, to_float_series(df[col_map['protein(g)']], 'Protein(g)')))
fat = dict(zip(foods, to_float_series(df[col_map['fat(g)']], 'Fat(g)')))
vitamin_c = dict(zip(foods, to_float_series(df[col_map['vitaminc(mg)']], 'VitaminC(mg)')))
cost = dict(zip(foods, to_float_series(df[col_map['cost']], 'Cost')))
m = gp.Model('DietOptimization')
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
        val = x_vars[f].X
        if val > 1e-05:
            print(f'{f}: {val:.3f} servings')
else:
    print(f'No optimal solution found. Status: {m.status}')