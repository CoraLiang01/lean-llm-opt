import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
foods = df['Food'].tolist()

def to_float_series(series, colname):
    try:
        return series.str.strip().astype(float)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{colname}' to float: {e}")
calories = dict(zip(foods, to_float_series(df['Calories'], 'Calories')))
protein = dict(zip(foods, to_float_series(df['Protein(g)'], 'Protein(g)')))
fat = dict(zip(foods, to_float_series(df['Fat(g)'], 'Fat(g)')))
vitamin_c = dict(zip(foods, to_float_series(df['VitaminC(mg)'], 'VitaminC(mg)')))
cost = dict(zip(foods, to_float_series(df['Cost'], 'Cost')))
for food in foods:
    for (param, param_dict) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
        if food not in param_dict:
            raise KeyError(f"Missing {param} data for food '{food}'.")
m = gp.Model('DietOptimization')
x_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * x_vars[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * x_vars[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[f] * x_vars[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[f] * x_vars[f] for f in foods)) >= 60, name='vitaminc_min')
m.addConstr(gp.quicksum((fat[f] * x_vars[f] for f in foods)) <= 70, name='fat_max')
m.optimize()