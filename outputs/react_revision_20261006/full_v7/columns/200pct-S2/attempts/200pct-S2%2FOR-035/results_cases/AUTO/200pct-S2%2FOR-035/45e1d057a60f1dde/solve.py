import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
required_columns = {'Food': str, 'Calories': float, 'Protein(g)': float, 'Fat(g)': float, 'VitaminC(mg)': float, 'Cost': float}
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
foods = df['Food'].tolist()
n_foods = len(foods)

def convert_column(colname, dtype):
    try:
        return df[colname].astype(dtype).values
    except Exception as e:
        raise ValueError(f"Error converting column '{colname}' to {dtype}: {e}")
calories = dict(zip(foods, convert_column('Calories', float)))
protein = dict(zip(foods, convert_column('Protein(g)', float)))
fat = dict(zip(foods, convert_column('Fat(g)', float)))
vitamin_c = dict(zip(foods, convert_column('VitaminC(mg)', float)))
cost = dict(zip(foods, convert_column('Cost', float)))
for food in foods:
    for (param, param_dict) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
        if food not in param_dict:
            raise ValueError(f"Missing {param} value for food '{food}'.")

def solve_nutrition_blending(foods, calories, protein, fat, vitamin_c, cost):
    m = gp.Model('DietBlending')
    m.Params.MIPGap = 0.0001
    servings_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[f] * servings_vars[f] for f in foods)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((calories[f] * servings_vars[f] for f in foods)) >= 2000, name='calories_min')
    m.addConstr(gp.quicksum((protein[f] * servings_vars[f] for f in foods)) >= 50, name='protein_min')
    m.addConstr(gp.quicksum((vitamin_c[f] * servings_vars[f] for f in foods)) >= 60, name='vitaminC_min')
    m.addConstr(gp.quicksum((fat[f] * servings_vars[f] for f in foods)) <= 70, name='fat_max')
    m.optimize()
    return m
m = solve_nutrition_blending(foods, calories, protein, fat, vitamin_c, cost)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X:.6f}')
else:
    print(f'Solver status: {m.status}')