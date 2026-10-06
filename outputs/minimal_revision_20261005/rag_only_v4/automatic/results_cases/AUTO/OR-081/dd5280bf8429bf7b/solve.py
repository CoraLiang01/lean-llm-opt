import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv', sep=',')

    def normalize_id(s):
        return ' '.join(str(s).split()).casefold()
    cost_df['Food_norm'] = cost_df['Food'].apply(normalize_id)
    foods = list(cost_df['Food'])
    calories = dict(zip(cost_df['Food'], cost_df['Calories']))
    protein = dict(zip(cost_df['Food'], cost_df['Protein(g)']))
    fat = dict(zip(cost_df['Food'], cost_df['Fat(g)']))
    vitamin_c = dict(zip(cost_df['Food'], cost_df['VitaminC(mg)']))
    cost = dict(zip(cost_df['Food'], cost_df['Cost']))
    for f in foods:
        if f not in calories or f not in protein or f not in fat or (f not in vitamin_c) or (f not in cost):
            raise ValueError(f'Missing parameter(s) for food: {f}')
    m = gp.Model('meal_plan')
    x = m.addVars(foods, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[f] * x[f] for f in foods)), GRB.MINIMIZE)
    m.addConstr(gp.quicksum((calories[f] * x[f] for f in foods)) >= 2000, name='cal_min')
    m.addConstr(gp.quicksum((protein[f] * x[f] for f in foods)) >= 50, name='prot_min')
    m.addConstr(gp.quicksum((vitamin_c[f] * x[f] for f in foods)) >= 60, name='vitc_min')
    m.addConstr(gp.quicksum((fat[f] * x[f] for f in foods)) <= 70, name='fat_max')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for f in foods:
            var = x[f]
            print(f'{var.VarName} {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()