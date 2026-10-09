import gurobipy as gp
import pandas as pd
import numpy as np
import re
item_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant18/inputs/item_demand.csv'
item_demand_df = pd.read_csv(item_demand_path, sep=',', dtype=str, keep_default_na=False)
if 'Item' not in item_demand_df.columns or 'Demand' not in item_demand_df.columns:
    raise KeyError("item_demand.csv must contain 'Item' and 'Demand' columns.")
item_demand_df['Item'] = item_demand_df['Item'].str.strip()
items = item_demand_df['Item'].tolist()
demands = item_demand_df.set_index('Item')['Demand'].astype(int).to_dict()
cutting_patterns_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant18/inputs/cutting_patterns.csv'
cutting_patterns_df = pd.read_csv(cutting_patterns_path, sep=',', dtype=str, keep_default_na=False)
if 'Pattern' not in cutting_patterns_df.columns:
    raise KeyError("cutting_patterns.csv must contain 'Pattern' column.")
cutting_patterns_df['Pattern'] = cutting_patterns_df['Pattern'].str.strip()
patterns = cutting_patterns_df['Pattern'].tolist()
for item in items:
    if item not in cutting_patterns_df.columns:
        raise KeyError(f"Item '{item}' from item_demand.csv not found as a column in cutting_patterns.csv.")
pieces_per_pattern = {item: {} for item in items}
for (idx, row) in cutting_patterns_df.iterrows():
    pattern = row['Pattern']
    for item in items:
        try:
            pieces = int(row[item])
        except ValueError:
            raise ValueError(f"Invalid integer value for item '{item}' in pattern '{pattern}'.")
        pieces_per_pattern[item][pattern] = pieces
m = gp.Model('CuttingStock_MinRolls')
y_vars = m.addVars(patterns, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((y_vars[p] for p in patterns)), gp.GRB.MINIMIZE)
for item in items:
    m.addConstr(gp.quicksum((pieces_per_pattern[item][p] * y_vars[p] for p in patterns)) >= demands[item], name=f'demand_{item}')
m.optimize()