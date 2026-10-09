import gurobipy as gp
import pandas as pd
import numpy as np
item_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant18/inputs/item_demand.csv'
cutting_patterns_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant18/inputs/cutting_patterns.csv'
df_demand = pd.read_csv(item_demand_path, sep=',')
df_demand['Item'] = df_demand['Item'].astype(str).str.strip()
items = df_demand['Item'].tolist()
item_set = set(items)
demand = dict(zip(df_demand['Item'], df_demand['Demand']))
df_patterns = pd.read_csv(cutting_patterns_path, sep=',')
df_patterns['Pattern'] = df_patterns['Pattern'].astype(str).str.strip()
patterns = df_patterns['Pattern'].tolist()
pattern_set = set(patterns)
pattern_item_cols = [col for col in df_patterns.columns if col in item_set]
if set(pattern_item_cols) != item_set:
    missing = item_set - set(pattern_item_cols)
    extra = set(pattern_item_cols) - item_set
    raise ValueError(f'Mismatch between items in demand and pattern columns. Missing in patterns: {missing}. Extra: {extra}')
pieces = {}
for p in patterns:
    pieces[p] = {}
    row = df_patterns.loc[df_patterns['Pattern'] == p]
    if row.empty:
        raise ValueError(f'Pattern {p} not found in cutting_patterns.csv')
    for i in items:
        val = row.iloc[0][i]
        if not (isinstance(val, (int, np.integer)) and val >= 0):
            raise ValueError(f'Invalid piece count for pattern {p}, item {i}: {val}')
        pieces[p][i] = int(val)
m = gp.Model('CuttingStock_MinRolls')
y = m.addVars(patterns, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((y[p] for p in patterns)), gp.GRB.MINIMIZE)
for i in items:
    m.addConstr(gp.quicksum((pieces[p][i] * y[p] for p in patterns)) >= demand[i], name=f'demand_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (minimum rolls used)')
    print('--- Pattern Usage ---')
    for p in patterns:
        val = y[p].X
        if val > 1e-06:
            print(f'Pattern {p}: {val:.0f} rolls')
    print('--- Demand Satisfaction ---')
    for i in items:
        produced = sum((pieces[p][i] * y[p].X for p in patterns))
        print(f'Item {i}: Demand = {demand[i]}, Produced = {int(round(produced))}')
else:
    print(f'No optimal solution found. Status: {m.status}')