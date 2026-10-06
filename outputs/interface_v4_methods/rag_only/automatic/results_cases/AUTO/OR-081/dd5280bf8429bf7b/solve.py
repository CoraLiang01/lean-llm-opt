import pandas as pd
import gurobipy as gp
from gurobipy import GRB
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv', sep=',')
cost_df['Food_key'] = cost_df['Food'].astype(str).str.strip()
Foods = cost_df['Food_key'].tolist()
calories = dict(zip(cost_df['Food_key'], cost_df['Calories'].astype(float)))
protein = dict(zip(cost_df['Food_key'], cost_df['Protein(g)'].astype(float)))
fat = dict(zip(cost_df['Food_key'], cost_df['Fat(g)'].astype(float)))
vitamin_c = dict(zip(cost_df['Food_key'], cost_df['VitaminC(mg)'].astype(float)))
cost = dict(zip(cost_df['Food_key'], cost_df['Cost'].astype(float)))
for f in Foods:
    if f not in calories or f not in protein or f not in fat or (f not in vitamin_c) or (f not in cost):
        raise ValueError(f'Missing parameter(s) for food: {f}')
m = gp.Model('OneDayMealPlan')
x = m.addVars(Foods, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * x[f] for f in Foods)), GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * x[f] for f in Foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[f] * x[f] for f in Foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[f] * x[f] for f in Foods)) >= 60, name='vitaminC_min')
m.addConstr(gp.quicksum((fat[f] * x[f] for f in Foods)) <= 70, name='fat_max')
m.optimize()