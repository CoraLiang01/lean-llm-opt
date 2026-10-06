import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S2/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',')

def normcol(s):
    return re.sub('\\s+', '', s.strip().casefold())
colmap = {normcol(c): c for c in df.columns}
required_cols = {'food': None, 'calories': None, 'protein(g)': None, 'fat(g)': None, 'vitaminc(mg)': None, 'cost': None}
for k in required_cols:
    found = [c for c in df.columns if normcol(c) == k]
    if not found:
        raise KeyError(f"Required column '{k}' not found in CSV columns: {list(df.columns)}")
    required_cols[k] = found[0]
foods = df[required_cols['food']].astype(str).tolist()
calories = df.set_index(required_cols['food'])[required_cols['calories']].astype(float).to_dict()
protein = df.set_index(required_cols['food'])[required_cols['protein(g)']].astype(float).to_dict()
fat = df.set_index(required_cols['food'])[required_cols['fat(g)']].astype(float).to_dict()
vitc = df.set_index(required_cols['food'])[required_cols['vitaminc(mg)']].astype(float).to_dict()
cost = df.set_index(required_cols['food'])[required_cols['cost']].astype(float).to_dict()
for f in foods:
    for pname, pdict in [('Calories', calories), ('Protein', protein), ('Fat', fat), ('VitaminC', vitc), ('Cost', cost)]:
        if f not in pdict:
            raise ValueError(f"Food '{f}' missing {pname} data.")
m = gp.Model('DietPlan')
x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * x[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * x[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[f] * x[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitc[f] * x[f] for f in foods)) >= 60, name='vitc_min')
m.addConstr(gp.quicksum((fat[f] * x[f] for f in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for f in foods:
        val = x[f].X
        if val > 1e-05:
            print(f'{f}: {val:.3f} servings')
    total_cal = sum((calories[f] * x[f].X for f in foods))
    total_prot = sum((protein[f] * x[f].X for f in foods))
    total_fat = sum((fat[f] * x[f].X for f in foods))
    total_vitc = sum((vitc[f] * x[f].X for f in foods))
    print('\n--- Achieved Nutrition Totals ---')
    print(f'Calories:   {total_cal:.1f} kcal')
    print(f'Protein:    {total_prot:.1f} g')
    print(f'Fat:        {total_fat:.1f} g')
    print(f'Vitamin C:  {total_vitc:.1f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')