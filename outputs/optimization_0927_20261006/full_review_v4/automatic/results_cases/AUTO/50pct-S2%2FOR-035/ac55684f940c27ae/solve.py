import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
foods = df['Food'].tolist()

def to_float_series(series):
    return series.apply(lambda x: float(x.strip()))
calories = to_float_series(df['Calories'])
protein = to_float_series(df['Protein(g)'])
fat = to_float_series(df['Fat(g)'])
vitamin_c = to_float_series(df['VitaminC(mg)'])
cost = to_float_series(df['Cost'])
calories_dict = dict(zip(df['Food'], calories))
protein_dict = dict(zip(df['Food'], protein))
fat_dict = dict(zip(df['Food'], fat))
vitamin_c_dict = dict(zip(df['Food'], vitamin_c))
cost_dict = dict(zip(df['Food'], cost))
m = gp.Model('OneDayMealPlan')
x_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost_dict[i] * x_vars[i] for i in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories_dict[i] * x_vars[i] for i in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein_dict[i] * x_vars[i] for i in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c_dict[i] * x_vars[i] for i in foods)) >= 60, name='vitaminC_min')
m.addConstr(gp.quicksum((fat_dict[i] * x_vars[i] for i in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for i in foods:
        val = x_vars[i].X
        if val > 1e-05:
            print(f'{i}: {val:.3f} servings')
    total_cal = sum((calories_dict[i] * x_vars[i].X for i in foods))
    total_prot = sum((protein_dict[i] * x_vars[i].X for i in foods))
    total_fat = sum((fat_dict[i] * x_vars[i].X for i in foods))
    total_vitc = sum((vitamin_c_dict[i] * x_vars[i].X for i in foods))
    print('\n--- Achieved Nutrient Totals ---')
    print(f'Calories:   {total_cal:.1f} kcal')
    print(f'Protein:    {total_prot:.1f} g')
    print(f'Fat:        {total_fat:.1f} g')
    print(f'Vitamin C:  {total_vitc:.1f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')