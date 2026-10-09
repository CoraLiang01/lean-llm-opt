import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
df.columns = [col.strip() for col in df.columns]
required_cols = ['Food', 'Calories', 'Protein(g)', 'Fat(g)', 'VitaminC(mg)', 'Cost']
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
foods = df['Food'].tolist()
calories = pd.to_numeric(df['Calories'], errors='raise')
protein = pd.to_numeric(df['Protein(g)'], errors='raise')
fat = pd.to_numeric(df['Fat(g)'], errors='raise')
vitamin_c = pd.to_numeric(df['VitaminC(mg)'], errors='raise')
cost = pd.to_numeric(df['Cost'], errors='raise')
calories_dict = dict(zip(foods, calories))
protein_dict = dict(zip(foods, protein))
fat_dict = dict(zip(foods, fat))
vitamin_c_dict = dict(zip(foods, vitamin_c))
cost_dict = dict(zip(foods, cost))
m = gp.Model('MealPlanMinCost')
x_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost_dict[i] * x_vars[i] for i in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories_dict[i] * x_vars[i] for i in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein_dict[i] * x_vars[i] for i in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c_dict[i] * x_vars[i] for i in foods)) >= 60, name='vitamin_c_min')
m.addConstr(gp.quicksum((fat_dict[i] * x_vars[i] for i in foods)) <= 70, name='fat_max')
m.optimize()