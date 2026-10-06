import gurobipy as gp
import pandas as pd
import numpy as np
import re
value_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others3/value.csv', sep=',')
required_cols = {'item', 'value', 'weight'}
if not required_cols.issubset(set(value_df.columns)):
    missing = required_cols - set(value_df.columns)
    raise ValueError(f'Missing required columns in value.csv: {missing}')
value_df['item'] = value_df['item'].astype(str)
items = list(value_df['item'].unique())
value_dict = dict(zip(value_df['item'], value_df['value']))
weight_dict = dict(zip(value_df['item'], value_df['weight']))
if set(items) != set(value_dict.keys()) or set(items) != set(weight_dict.keys()):
    raise ValueError('Mismatch in items and value/weight keys.')

def solve_knapsack(items, value_dict, weight_dict, weight_limit):
    m = gp.Model('Knapsack')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((value_dict[i] * x[i] for i in items)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight_dict[i] * x[i] for i in items)) <= weight_limit, name='weight_limit')
    m.optimize()
    return (m, x)
(m, x) = solve_knapsack(items, value_dict, weight_dict, 15)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in items:
        print(f'{x[i].VarName} {x[i].X}')
else:
    print(f'Solver status: {m.status}')