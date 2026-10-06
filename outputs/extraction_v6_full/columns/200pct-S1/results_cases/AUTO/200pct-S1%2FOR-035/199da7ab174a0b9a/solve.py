import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S1/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',')
foods = df['Food'].astype(str).tolist()
calories = df.set_index('Food')['Calories'].astype(float).to_dict()
protein = df.set_index('Food')['Protein(g)'].astype(float).to_dict()
fat = df.set_index('Food')['Fat(g)'].astype(float).to_dict()
vitamin_c = df.set_index('Food')['VitaminC(mg)'].astype(float).to_dict()
cost = df.set_index('Food')['Cost'].astype(float).to_dict()
for f in foods:
    for param, d in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
        if f not in d:
            raise ValueError(f"Missing {param} data for food '{f}'.")

def solve_meal_plan(foods, calories, protein, fat, vitamin_c, cost):
    m = gp.Model('OneDayMealPlan')
    x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='servings')
    m.setObjective(gp.quicksum((cost[f] * x[f] for f in foods)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((calories[f] * x[f] for f in foods)) >= 2000, name='min_calories')
    m.addConstr(gp.quicksum((protein[f] * x[f] for f in foods)) >= 50, name='min_protein')
    m.addConstr(gp.quicksum((vitamin_c[f] * x[f] for f in foods)) >= 60, name='min_vitamin_c')
    m.addConstr(gp.quicksum((fat[f] * x[f] for f in foods)) <= 70, name='max_fat')
    m.optimize()
    return m
m = solve_meal_plan(foods, calories, protein, fat, vitamin_c, cost)