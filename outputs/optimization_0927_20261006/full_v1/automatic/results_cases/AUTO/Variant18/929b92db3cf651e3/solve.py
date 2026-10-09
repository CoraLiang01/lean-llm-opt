import gurobipy as gp
import pandas as pd
import numpy as np
item_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant18/inputs/item_demand.csv'
cutting_patterns_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant18/inputs/cutting_patterns.csv'
item_demand_df = pd.read_csv(item_demand_path, sep=',', dtype=str, keep_default_na=False)
cutting_patterns_df = pd.read_csv(cutting_patterns_path, sep=',', dtype=str, keep_default_na=False)
items = item_demand_df['Item'].astype(str).str.strip().tolist()
patterns = cutting_patterns_df['Pattern'].astype(str).str.strip().tolist()
demand = {}
for (idx, row) in item_demand_df.iterrows():
    item = str(row['Item']).strip()
    try:
        demand[item] = int(row['Demand'])
    except Exception:
        raise ValueError(f"Invalid demand value for item '{item}': {row['Demand']}")
for item in items:
    if item not in cutting_patterns_df.columns:
        raise KeyError(f"Item '{item}' from item_demand.csv not found as a column in cutting_patterns.csv")
pattern_item_pieces = {}
for (idx, row) in cutting_patterns_df.iterrows():
    pattern = str(row['Pattern']).strip()
    for item in items:
        try:
            pattern_item_pieces[pattern, item] = int(row[item])
        except Exception:
            raise ValueError(f"Invalid piece count for pattern '{pattern}', item '{item}': {row[item]}")
m = gp.Model('CuttingStock_MinRolls')
y_vars = m.addVars(patterns, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((y_vars[p] for p in patterns)), gp.GRB.MINIMIZE)
for item in items:
    m.addConstr(gp.quicksum((pattern_item_pieces[p, item] * y_vars[p] for p in patterns)) >= demand[item], name=f'demand_{item}')
m.optimize()