import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
foods = df['Food'].tolist()

def to_float_series(series):
    return series.str.strip().astype(float)
calories = to_float_series(df['Calories'])
protein = to_float_series(df['Protein(g)'])
fat = to_float_series(df['Fat(g)'])
vitamin_c = to_float_series(df['VitaminC(mg)'])
cost = to_float_series(df['Cost'])
calories_dict = dict(zip(foods, calories))
protein_dict = dict(zip(foods, protein))
fat_dict = dict(zip(foods, fat))
vitamin_c_dict = dict(zip(foods, vitamin_c))
cost_dict = dict(zip(foods, cost))
m = gp.Model('DietProblem')
servings_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost_dict[f] * servings_vars[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories_dict[f] * servings_vars[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein_dict[f] * servings_vars[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c_dict[f] * servings_vars[f] for f in foods)) >= 60, name='vitaminC_min')
m.addConstr(gp.quicksum((fat_dict[f] * servings_vars[f] for f in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for f in foods:
        val = servings_vars[f].X
        if val > 1e-05:
            print(f'{f}: {val:.3f} servings')
else:
    print(f'No optimal solution found. Status: {m.status}')