import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',')
foods = df['Food'].astype(str).tolist()

def get_param_dict(colname, dtype=float):
    s = df.set_index('Food')[colname]
    if s.isnull().any():
        missing = s[s.isnull()].index.tolist()
        raise ValueError(f'Missing values for {colname} in foods: {missing}')
    return s.astype(dtype).to_dict()
calories = get_param_dict('Calories', dtype=float)
protein = get_param_dict('Protein(g)', dtype=float)
fat = get_param_dict('Fat(g)', dtype=float)
vitaminc = get_param_dict('VitaminC(mg)', dtype=float)
cost = get_param_dict('Cost', dtype=float)
for food in foods:
    for (pname, pdict) in [('Calories', calories), ('Protein(g)', protein), ('Fat(g)', fat), ('VitaminC(mg)', vitaminc), ('Cost', cost)]:
        if food not in pdict:
            raise ValueError(f"Food '{food}' missing parameter '{pname}'.")
m = gp.Model('DietPlan')
x = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i] * x[i] for i in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[i] * x[i] for i in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[i] * x[i] for i in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitaminc[i] * x[i] for i in foods)) >= 60, name='vitaminc_min')
m.addConstr(gp.quicksum((fat[i] * x[i] for i in foods)) <= 70, name='fat_max')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X:.6f}')
else:
    print(f'Solver status: {m.status}')