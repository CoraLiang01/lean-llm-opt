import pandas as pd
import gurobipy as gp
from gurobipy import GRB
value_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others3/value.csv', sep=',')
value_df['item'] = value_df['item'].astype(int)
items = value_df['item'].unique().tolist()
value_dict = dict(zip(value_df['item'], value_df['value']))
weight_dict = dict(zip(value_df['item'], value_df['weight']))
if set(items) != set(value_dict.keys()) or set(items) != set(weight_dict.keys()):
    raise ValueError('Mismatch in item indices between value and weight data.')
m = gp.Model('shopping_centre_knapsack')
x = m.addVars(items, vtype=GRB.BINARY, lb=0, ub=1, name='')
m.setObjective(gp.quicksum((value_dict[i] * x[i] for i in items)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x[i] for i in items)) <= 15, name='weight_limit')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in items:
        var = x[i]
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.Status}')