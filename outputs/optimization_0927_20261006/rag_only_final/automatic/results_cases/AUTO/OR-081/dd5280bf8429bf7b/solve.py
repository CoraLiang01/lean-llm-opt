import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv', dtype=str, keep_default_na=False)
foods = cost_df['Food'].astype(str).tolist()

def to_float(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Column '{colname}' could not be converted to float: {e}")
calories = dict(zip(foods, to_float(cost_df['Calories'], 'Calories')))
protein = dict(zip(foods, to_float(cost_df['Protein(g)'], 'Protein(g)')))
fat = dict(zip(foods, to_float(cost_df['Fat(g)'], 'Fat(g)')))
vitamin_c = dict(zip(foods, to_float(cost_df['VitaminC(mg)'], 'VitaminC(mg)')))
cost = dict(zip(foods, to_float(cost_df['Cost'], 'Cost')))
for food in foods:
    for (param, param_dict) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
        if food not in param_dict:
            raise ValueError(f"Missing {param} data for food '{food}'.")
m = Model('meal_plan')
servings_vars = m.addVars(foods, lb=0.0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(quicksum((cost[i] * servings_vars[i] for i in foods)), GRB.MINIMIZE)
m.addConstr(quicksum((calories[i] * servings_vars[i] for i in foods)) >= 2000, name='calories_min')
m.addConstr(quicksum((protein[i] * servings_vars[i] for i in foods)) >= 50, name='protein_min')
m.addConstr(quicksum((vitamin_c[i] * servings_vars[i] for i in foods)) >= 60, name='vitamin_c_min')
m.addConstr(quicksum((fat[i] * servings_vars[i] for i in foods)) <= 70, name='fat_max')
m.optimize()