import gurobipy as gp
import pandas as pd
import numpy as np
item_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv'
cutting_patterns_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv'
item_demand_df = pd.read_csv(item_demand_path, sep=',')
item_demand_df['Item'] = item_demand_df['Item'].astype(str).str.strip()
items = list(item_demand_df['Item'].unique())
demand = dict(zip(item_demand_df['Item'], item_demand_df['Demand']))
patterns_df = pd.read_csv(cutting_patterns_path, sep=',')
patterns_df['Pattern'] = patterns_df['Pattern'].astype(str).str.strip()
patterns = list(patterns_df['Pattern'].unique())
missing_items = [i for i in items if i not in patterns_df.columns]
if missing_items:
    raise ValueError(f'Missing item columns in cutting_patterns.csv: {missing_items}')
units_produced = {}
for p in patterns:
    row = patterns_df.loc[patterns_df['Pattern'] == p]
    if row.empty:
        raise ValueError(f'Pattern {p} not found in cutting_patterns.csv')
    units_produced[p] = {}
    for i in items:
        val = row.iloc[0][i]
        if not np.issubdtype(type(val), np.integer):
            raise ValueError(f'Non-integer value for pattern {p}, item {i}: {val}')
        units_produced[p][i] = int(val)
m = gp.Model('CuttingStockPatternSelection')
y = m.addVars(patterns, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((y[p] for p in patterns)), gp.GRB.MINIMIZE)
for i in items:
    m.addConstr(gp.quicksum((units_produced[p][i] * y[p] for p in patterns)) >= demand[i], name=f'demand_{i}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for p in patterns:
        print(f'{y[p].VarName} {y[p].X}')
else:
    print(f'Solver status: {m.status}')