import gurobipy as gp
import pandas as pd
import numpy as np
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv', sep=',')
foods = cost_df['Food'].astype(str).tolist()
calories = dict(zip(cost_df['Food'].astype(str), cost_df['Calories'].astype(float)))
protein = dict(zip(cost_df['Food'].astype(str), cost_df['Protein(g)'].astype(float)))
fat = dict(zip(cost_df['Food'].astype(str), cost_df['Fat(g)'].astype(float)))
vitamin_c = dict(zip(cost_df['Food'].astype(str), cost_df['VitaminC(mg)'].astype(float)))
cost = dict(zip(cost_df['Food'].astype(str), cost_df['Cost'].astype(float)))
m = gp.Model('MealPlan')
x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i] * x[i] for i in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[i] * x[i] for i in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[i] * x[i] for i in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[i] * x[i] for i in foods)) >= 60, name='vitaminC_min')
m.addConstr(gp.quicksum((fat[i] * x[i] for i in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for i in foods:
        if x[i].X > 1e-05:
            print(f'{i}: {x[i].X:.3f} servings')
    total_cal = sum((calories[i] * x[i].X for i in foods))
    total_prot = sum((protein[i] * x[i].X for i in foods))
    total_fat = sum((fat[i] * x[i].X for i in foods))
    total_vitc = sum((vitamin_c[i] * x[i].X for i in foods))
    print('\n--- Achieved Nutrition Totals ---')
    print(f'Calories:   {total_cal:.1f} kcal')
    print(f'Protein:    {total_prot:.1f} g')
    print(f'Fat:        {total_fat:.1f} g')
    print(f'Vitamin C:  {total_vitc:.1f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')