import gurobipy as gp
import pandas as pd
import numpy as np
value_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others3/value.csv', sep=',')
value_df['item'] = value_df['item'].astype(int)
value_df['value'] = value_df['value'].astype(int)
value_df['weight'] = value_df['weight'].astype(int)
items = value_df['item'].tolist()
value_dict = dict(zip(value_df['item'], value_df['value']))
weight_dict = dict(zip(value_df['item'], value_df['weight']))
if set(items) != set(value_dict.keys()) or set(items) != set(weight_dict.keys()):
    raise ValueError('Mismatch in item indices between value and weight columns.')
m = gp.Model('KnapsackSelection')
x = m.addVars(items, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[i] * x[i] for i in items)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x[i] for i in items)) <= 15, name='weight_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.0f}')
    print('Selected items:')
    for i in items:
        if x[i].X > 0.5:
            print(f'  Item {i}: value={value_dict[i]}, weight={weight_dict[i]}')
else:
    print(f'No optimal solution found. Status: {m.status}')