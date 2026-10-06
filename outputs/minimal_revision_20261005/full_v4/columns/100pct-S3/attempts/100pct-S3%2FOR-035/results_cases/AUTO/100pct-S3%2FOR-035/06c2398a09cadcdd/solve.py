import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',')
foods = df['Food'].astype(str).tolist()

def get_param_dict(colname, foods):
    s = df.set_index('Food')[colname]
    if not set(foods).issubset(set(s.index)):
        missing = set(foods) - set(s.index)
        raise ValueError(f'Missing parameter values for foods: {missing}')
    return s.astype(float).to_dict()
calories = get_param_dict('Calories', foods)
protein = get_param_dict('Protein(g)', foods)
fat = get_param_dict('Fat(g)', foods)
vitamin_c = get_param_dict('VitaminC(mg)', foods)
cost = get_param_dict('Cost', foods)
for (param, d) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitamin_c), ('Cost', cost)]:
    if set(d.keys()) != set(foods):
        raise ValueError(f'Parameter {param} missing values for some foods.')
m = gp.Model('DietProblem')
x = m.addVars(foods, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i] * x[i] for i in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[i] * x[i] for i in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[i] * x[i] for i in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[i] * x[i] for i in foods)) >= 60, name='vitamin_c_min')
m.addConstr(gp.quicksum((fat[i] * x[i] for i in foods)) <= 70, name='fat_max')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for i in foods:
        var = x[i]
        print(f'{var.VarName} {var.X:.6f}')
else:
    print(f'Solver status: {m.status}')