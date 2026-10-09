import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
df['Food'] = df['Food'].astype(str)
for col in ['Calories', 'Protein(g)', 'Fat(g)', 'VitaminC(mg)', 'Cost']:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in cost.csv")
    df[col] = pd.to_numeric(df[col], errors='raise')
foods = df['Food'].tolist()
calories = dict(zip(df['Food'], df['Calories']))
protein = dict(zip(df['Food'], df['Protein(g)']))
fat = dict(zip(df['Food'], df['Fat(g)']))
vitc = dict(zip(df['Food'], df['VitaminC(mg)']))
cost = dict(zip(df['Food'], df['Cost']))
m = gp.Model('DietProblem')
servings_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * servings_vars[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * servings_vars[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[f] * servings_vars[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitc[f] * servings_vars[f] for f in foods)) >= 60, name='vitc_min')
m.addConstr(gp.quicksum((fat[f] * servings_vars[f] for f in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for f in foods:
        val = servings_vars[f].X
        if val > 1e-05:
            print(f'{f}: {val:.4f} servings')
    total_cal = sum((calories[f] * servings_vars[f].X for f in foods))
    total_prot = sum((protein[f] * servings_vars[f].X for f in foods))
    total_fat = sum((fat[f] * servings_vars[f].X for f in foods))
    total_vitc = sum((vitc[f] * servings_vars[f].X for f in foods))
    print('\n--- Achieved Nutrient Totals ---')
    print(f'Calories: {total_cal:.2f} kcal')
    print(f'Protein: {total_prot:.2f} g')
    print(f'Fat: {total_fat:.2f} g')
    print(f'Vitamin C: {total_vitc:.2f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')