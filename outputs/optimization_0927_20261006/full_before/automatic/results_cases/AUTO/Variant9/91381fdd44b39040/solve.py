import gurobipy as gp
import pandas as pd
import numpy as np
import re
item_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv'
cutting_patterns_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv'
item_demand_df = pd.read_csv(item_demand_path, sep=',')
item_demand_df['Item'] = item_demand_df['Item'].astype(str).str.strip()
items = item_demand_df['Item'].tolist()
demand = dict(zip(item_demand_df['Item'], item_demand_df['Demand']))
patterns_df = pd.read_csv(cutting_patterns_path, sep=',')
patterns_df['Pattern'] = patterns_df['Pattern'].astype(str).str.strip()
patterns = patterns_df['Pattern'].tolist()
missing_items = [i for i in items if i not in patterns_df.columns]
if missing_items:
    raise ValueError(f'Missing item columns in cutting_patterns.csv: {missing_items}')
pattern_item_qty = {}
for p in patterns:
    pattern_item_qty[p] = {}
    row = patterns_df.loc[patterns_df['Pattern'] == p]
    if row.empty:
        raise ValueError(f"Pattern '{p}' not found in cutting_patterns.csv")
    for i in items:
        val = row.iloc[0][i]
        if not (isinstance(val, (int, np.integer)) and val >= 0):
            raise ValueError(f"Invalid or missing value for pattern '{p}', item '{i}'")
        pattern_item_qty[p][i] = int(val)
m = gp.Model('CuttingStockPatternSelection')
y = m.addVars(patterns, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((y[p] for p in patterns)), gp.GRB.MINIMIZE)
for i in items:
    m.addConstr(gp.quicksum((pattern_item_qty[p][i] * y[p] for p in patterns)) >= demand[i], name=f'demand_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total number of standard rolls used: {int(round(m.objVal))}')
    print('--- Pattern Usage ---')
    for p in patterns:
        val = y[p].X
        if val > 1e-06:
            print(f'Pattern {p}: {int(round(val))} rolls')
    print('--- Demand Satisfaction ---')
    for i in items:
        produced = sum((pattern_item_qty[p][i] * y[p].X for p in patterns))
        print(f'Item {i}: Demand = {demand[i]}, Produced = {int(round(produced))}')
else:
    print(f'No optimal solution found. Status: {m.status}')