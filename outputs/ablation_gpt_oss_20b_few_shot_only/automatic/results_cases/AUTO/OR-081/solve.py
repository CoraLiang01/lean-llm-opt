import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    csv_path = 'Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv'
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    required_cols = ['Food', 'Calories', 'Protein(g)', 'Fat(g)', 'VitaminC(mg)', 'Cost']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in CSV")
    foods = df['Food'].tolist()
    calories = {}
    protein = {}
    fat = {}
    vitaminC = {}
    cost = {}
    for (_, row) in df.iterrows():
        f = row['Food']
        try:
            calories[f] = float(row['Calories'])
            protein[f] = float(row['Protein(g)'])
            fat[f] = float(row['Fat(g)'])
            vitaminC[f] = float(row['VitaminC(mg)'])
            cost[f] = float(row['Cost'])
        except ValueError as e:
            raise ValueError(f"Invalid numeric value for food '{f}': {e}")
    for f in foods:
        if f not in calories or f not in protein or f not in fat or (f not in vitaminC) or (f not in cost):
            raise ValueError(f"Incomplete data for food '{f}'")
    m = gurobipy.Model('NutritionPlan')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(foods, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((cost[f] * x[f] for f in foods)), GRB.MINIMIZE)
    m.addConstr(gurobipy.quicksum((calories[f] * x[f] for f in foods)) >= 2000, name='Calories')
    m.addConstr(gurobipy.quicksum((protein[f] * x[f] for f in foods)) >= 50, name='Protein')
    m.addConstr(gurobipy.quicksum((vitaminC[f] * x[f] for f in foods)) >= 60, name='VitaminC')
    m.addConstr(gurobipy.quicksum((fat[f] * x[f] for f in foods)) <= 70, name='Fat')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'Optimal objective value: {m.ObjVal:.4f}')
        for var in m.getVars():
            print(f'{var.VarName} = {var.X:.4f}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()