import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
food_ids = df['Food'].tolist()

def to_float_series(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Column '{colname}' could not be fully converted to float: {e}")
calories = to_float_series(df['Calories'], 'Calories')
protein = to_float_series(df['Protein(g)'], 'Protein(g)')
fat = to_float_series(df['Fat(g)'], 'Fat(g)')
vitamin_c = to_float_series(df['VitaminC(mg)'], 'VitaminC(mg)')
cost = to_float_series(df['Cost'], 'Cost')
calories_dict = dict(zip(food_ids, calories))
protein_dict = dict(zip(food_ids, protein))
fat_dict = dict(zip(food_ids, fat))
vitamin_c_dict = dict(zip(food_ids, vitamin_c))
cost_dict = dict(zip(food_ids, cost))
m = gp.Model('MealPlanMinCost')
servings_vars = m.addVars(food_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost_dict[i] * servings_vars[i] for i in food_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories_dict[i] * servings_vars[i] for i in food_ids)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein_dict[i] * servings_vars[i] for i in food_ids)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c_dict[i] * servings_vars[i] for i in food_ids)) >= 60, name='vitamin_c_min')
m.addConstr(gp.quicksum((fat_dict[i] * servings_vars[i] for i in food_ids)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for i in food_ids:
        val = servings_vars[i].X
        if val > 1e-05:
            print(f'{i}: {val:.4f} servings')
else:
    print(f'No optimal solution found. Status: {m.status}')