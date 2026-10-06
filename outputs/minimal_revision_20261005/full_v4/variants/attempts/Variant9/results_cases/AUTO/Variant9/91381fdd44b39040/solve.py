import gurobipy as gp
import pandas as pd
import numpy as np
import re
item_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv'
cutting_patterns_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv'
item_demand_df = pd.read_csv(item_demand_path, sep=',')
if item_demand_df['Item'].isnull().any() or item_demand_df['Demand'].isnull().any():
    raise ValueError('Missing values in item_demand.csv')
items = item_demand_df['Item'].astype(str).tolist()
demands = dict(zip(item_demand_df['Item'].astype(str), item_demand_df['Demand'].astype(int)))
patterns_df = pd.read_csv(cutting_patterns_path, sep=',')
if patterns_df['Pattern'].isnull().any():
    raise ValueError('Missing Pattern values in cutting_patterns.csv')
patterns = patterns_df['Pattern'].astype(str).tolist()
for item in items:
    if item not in patterns_df.columns:
        raise KeyError(f"Item '{item}' not found as a column in cutting_patterns.csv")
units = {}
for (idx, row) in patterns_df.iterrows():
    p = str(row['Pattern'])
    for i in items:
        val = row[i]
        if not np.issubdtype(type(val), np.integer):
            raise ValueError(f'Non-integer value for pattern {p}, item {i}')
        units[i, p] = int(val)
for i in items:
    for p in patterns:
        if (i, p) not in units:
            raise KeyError(f'Missing units for item {i}, pattern {p}')
m = gp.Model('CuttingStockPatternSelection')
y = m.addVars(patterns, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((y[p] for p in patterns)), gp.GRB.MINIMIZE)
for i in items:
    m.addConstr(gp.quicksum((units[i, p] * y[p] for p in patterns)) >= demands[i], name=f'demand_{i}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for p in patterns:
        print(f'{y[p].VarName} {y[p].X}')
else:
    print(f'Solver status: {m.status}')