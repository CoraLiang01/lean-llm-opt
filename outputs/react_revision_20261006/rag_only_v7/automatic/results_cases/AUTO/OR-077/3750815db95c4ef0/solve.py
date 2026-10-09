import pandas as pd
import gurobipy as gp
from gurobipy import GRB
value_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others3/value.csv', dtype=str, keep_default_na=False)
try:
    value_df['item'] = value_df['item'].astype(int)
    value_df['value'] = value_df['value'].astype(int)
    value_df['weight'] = value_df['weight'].astype(int)
except Exception as e:
    raise ValueError(f'Error converting columns to int: {e}')
item_ids = value_df['item'].tolist()
value_dict = dict(zip(value_df['item'], value_df['value']))
weight_dict = dict(zip(value_df['item'], value_df['weight']))
if set(item_ids) != set(value_dict.keys()) or set(item_ids) != set(weight_dict.keys()):
    raise ValueError('Mismatch in item identifiers between index set and parameter dictionaries.')
WEIGHT_LIMIT = 15
m = gp.Model('shopping_centre_knapsack')
x_vars = m.addVars(item_ids, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[i] * x_vars[i] for i in item_ids)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i] for i in item_ids)) <= WEIGHT_LIMIT, name='weight_limit')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in item_ids:
        var = x_vars[i]
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.Status}')