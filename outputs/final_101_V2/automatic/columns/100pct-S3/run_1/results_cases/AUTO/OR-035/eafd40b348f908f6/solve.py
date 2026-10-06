import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S3/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',')
foods = df['Food'].astype(str).tolist()
calories = df.set_index('Food')['Calories'].astype(float).to_dict()
protein = df.set_index('Food')['Protein(g)'].astype(float).to_dict()
fat = df.set_index('Food')['Fat(g)'].astype(float).to_dict()
vitamin_c = df.set_index('Food')['VitaminC(mg)'].astype(float).to_dict()
cost = df.set_index('Food')['Cost'].astype(float).to_dict()
for food in foods:
    for param, d in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
        if food not in d:
            raise ValueError(f"Missing {param} for food '{food}'")
m = gp.Model('DietPlan')
x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i] * x[i] for i in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[i] * x[i] for i in foods)) >= 2000, name='cal_min')
m.addConstr(gp.quicksum((protein[i] * x[i] for i in foods)) >= 50, name='prot_min')
m.addConstr(gp.quicksum((vitamin_c[i] * x[i] for i in foods)) >= 60, name='vitc_min')
m.addConstr(gp.quicksum((fat[i] * x[i] for i in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for i in foods:
        if x[i].X > 1e-05:
            print(f'{i}: {x[i].X:.4f} servings')
    total_cal = sum((calories[i] * x[i].X for i in foods))
    total_prot = sum((protein[i] * x[i].X for i in foods))
    total_fat = sum((fat[i] * x[i].X for i in foods))
    total_vitc = sum((vitamin_c[i] * x[i].X for i in foods))
    print('\n--- Nutrient Totals ---')
    print(f'Calories:   {total_cal:.2f} kcal')
    print(f'Protein:    {total_prot:.2f} g')
    print(f'Fat:        {total_fat:.2f} g')
    print(f'Vitamin C:  {total_vitc:.2f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')