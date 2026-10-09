import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
required_columns = {'Food': None, 'Calories': None, 'Protein(g)': None, 'Fat(g)': None, 'VitaminC(mg)': None, 'Cost': None}

def find_col(df, pattern):
    for col in df.columns:
        if re.sub('\\s+', '', col).casefold() == re.sub('\\s+', '', pattern).casefold():
            return col
    raise KeyError(f"Could not find a column matching '{pattern}'")
for key in required_columns:
    required_columns[key] = find_col(df, key)
foods = df[required_columns['Food']].tolist()

def to_float_series(series):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f'Failed to convert series to float: {e}')
calories = dict(zip(foods, to_float_series(df[required_columns['Calories']])))
protein = dict(zip(foods, to_float_series(df[required_columns['Protein(g)']])))
fat = dict(zip(foods, to_float_series(df[required_columns['Fat(g)']])))
vitamin_c = dict(zip(foods, to_float_series(df[required_columns['VitaminC(mg)']])))
cost = dict(zip(foods, to_float_series(df[required_columns['Cost']])))
for food in foods:
    for (param, d) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
        if food not in d:
            raise KeyError(f"Missing {param} for food '{food}'")
m = gp.Model('MealPlanMinCost')
servings_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[food] * servings_vars[food] for food in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[food] * servings_vars[food] for food in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[food] * servings_vars[food] for food in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[food] * servings_vars[food] for food in foods)) >= 60, name='vitaminC_min')
m.addConstr(gp.quicksum((fat[food] * servings_vars[food] for food in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: ${m.objVal:.2f}')
    print('--- Meal Plan (servings per food) ---')
    for food in foods:
        val = servings_vars[food].X
        if val > 1e-05:
            print(f'{food}: {val:.3f} servings')
else:
    print(f'No optimal solution found. Status: {m.status}')