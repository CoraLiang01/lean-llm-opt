import gurobipy as gp
import pandas as pd
import numpy as np
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv', sep=',')
if cost_df['Food'].isnull().any():
    raise ValueError('Missing Food identifiers in cost.csv')
foods = cost_df['Food'].astype(str).tolist()
n_foods = len(foods)
calories = dict(zip(foods, cost_df['Calories'].astype(float)))
protein = dict(zip(foods, cost_df['Protein(g)'].astype(float)))
fat = dict(zip(foods, cost_df['Fat(g)'].astype(float)))
vitamin_c = dict(zip(foods, cost_df['VitaminC(mg)'].astype(float)))
cost = dict(zip(foods, cost_df['Cost'].astype(float)))
for f in foods:
    for param, d in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
        if f not in d or pd.isnull(d[f]):
            raise ValueError(f"Missing value for {param} in food '{f}'")
m = gp.Model('OneDayMealPlan')
x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * x[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * x[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[f] * x[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[f] * x[f] for f in foods)) >= 60, name='vitaminc_min')
m.addConstr(gp.quicksum((fat[f] * x[f] for f in foods)) <= 70, name='fat_max')
m.optimize()