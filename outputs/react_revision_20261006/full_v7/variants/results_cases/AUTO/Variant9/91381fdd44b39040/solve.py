import gurobipy as gp
import pandas as pd
import numpy as np
item_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv'
item_demand_df = pd.read_csv(item_demand_path, sep=',', dtype=str, keep_default_na=False)
item_demand_df['Item'] = item_demand_df['Item'].str.strip()
item_demand_df['Demand'] = item_demand_df['Demand'].astype(int)
items = item_demand_df['Item'].tolist()
item_to_demand = dict(zip(item_demand_df['Item'], item_demand_df['Demand']))
cutting_patterns_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv'
cutting_patterns_df = pd.read_csv(cutting_patterns_path, sep=',', dtype=str, keep_default_na=False)
cutting_patterns_df['Pattern'] = cutting_patterns_df['Pattern'].str.strip()
patterns = cutting_patterns_df['Pattern'].tolist()
for item in items:
    if item not in cutting_patterns_df.columns:
        raise ValueError(f"Item '{item}' from item_demand.csv not found as a column in cutting_patterns.csv.")
pattern_item_to_units = {}
for (_, row) in cutting_patterns_df.iterrows():
    pattern = row['Pattern']
    for item in items:
        try:
            units = int(row[item])
        except Exception as e:
            raise ValueError(f"Invalid or missing value for pattern '{pattern}', item '{item}': {row[item]}")
        pattern_item_to_units[pattern, item] = units
for pattern in patterns:
    for item in items:
        if (pattern, item) not in pattern_item_to_units:
            raise ValueError(f"Missing pattern-item coefficient for pattern '{pattern}', item '{item}'.")
m = gp.Model('CuttingStockPatternSelection')
pattern_vars = m.addVars(patterns, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((pattern_vars[p] for p in patterns)), gp.GRB.MINIMIZE)
for item in items:
    m.addConstr(gp.quicksum((pattern_item_to_units[p, item] * pattern_vars[p] for p in patterns)) >= item_to_demand[item], name=f'demand_{item}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for p in patterns:
        print(f'{pattern_vars[p].VarName} {pattern_vars[p].X}')
else:
    print(f'Solver status: {m.status}')