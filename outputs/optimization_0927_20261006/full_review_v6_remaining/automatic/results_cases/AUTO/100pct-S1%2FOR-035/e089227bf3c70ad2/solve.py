import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['Food', 'Calories', 'Protein(g)', 'Fat(g)', 'VitaminC(mg)', 'Cost']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in cost.csv")
foods = df['Food'].tolist()

def to_float_series(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Column '{colname}' could not be fully converted to float: {e}")
calories = dict(zip(foods, to_float_series(df['Calories'], 'Calories')))
protein = dict(zip(foods, to_float_series(df['Protein(g)'], 'Protein(g)')))
fat = dict(zip(foods, to_float_series(df['Fat(g)'], 'Fat(g)')))
vitamin_c = dict(zip(foods, to_float_series(df['VitaminC(mg)'], 'VitaminC(mg)')))
cost = dict(zip(foods, to_float_series(df['Cost'], 'Cost')))
for food in foods:
    for (param, d) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
        if food not in d:
            raise KeyError(f"Food '{food}' missing parameter '{param}'")
m = gp.Model('OneDayMealPlan')
servings_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * servings_vars[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * servings_vars[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[f] * servings_vars[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[f] * servings_vars[f] for f in foods)) >= 60, name='vitamin_c_min')
m.addConstr(gp.quicksum((fat[f] * servings_vars[f] for f in foods)) <= 70, name='fat_max')
m.optimize()