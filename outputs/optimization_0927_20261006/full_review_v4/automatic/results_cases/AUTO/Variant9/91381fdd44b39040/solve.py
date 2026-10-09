import gurobipy as gp
import pandas as pd
import numpy as np
import re
item_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv'
cutting_patterns_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv'
item_demand_df = pd.read_csv(item_demand_path, dtype=str, keep_default_na=False)
item_demand_df['Item'] = item_demand_df['Item'].str.strip()
item_demand_df['Demand'] = item_demand_df['Demand'].astype(int)
items = item_demand_df['Item'].tolist()
item_to_demand = dict(zip(item_demand_df['Item'], item_demand_df['Demand']))
cutting_patterns_df = pd.read_csv(cutting_patterns_path, dtype=str, keep_default_na=False)
cutting_patterns_df['Pattern'] = cutting_patterns_df['Pattern'].str.strip()
patterns = cutting_patterns_df['Pattern'].tolist()
for item in items:
    if item not in cutting_patterns_df.columns:
        raise KeyError(f"Item '{item}' not found as a column in cutting_patterns.csv")
    cutting_patterns_df[item] = cutting_patterns_df[item].astype(int)
pattern_item_qty = {pattern: {item: int(cutting_patterns_df.loc[cutting_patterns_df['Pattern'] == pattern, item].values[0]) for item in items} for pattern in patterns}
m = gp.Model('CuttingStockPatternSelection')
y_vars = m.addVars(patterns, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((y_vars[p] for p in patterns)), gp.GRB.MINIMIZE)
for item in items:
    m.addConstr(gp.quicksum((pattern_item_qty[p][item] * y_vars[p] for p in patterns)) >= item_to_demand[item], name=f'demand_{item}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total number of standard rolls used: {m.objVal:.0f}')
    print('--- Pattern Usage ---')
    for p in patterns:
        qty = y_vars[p].X
        if qty > 1e-06:
            print(f'Pattern {p}: {qty:.0f} times')
    print('--- Demand Satisfaction ---')
    for item in items:
        produced = sum((pattern_item_qty[p][item] * y_vars[p].X for p in patterns))
        print(f'Item {item}: Demand = {item_to_demand[item]}, Produced = {int(round(produced))}')
else:
    print(f'No optimal solution found. Status: {m.status}')