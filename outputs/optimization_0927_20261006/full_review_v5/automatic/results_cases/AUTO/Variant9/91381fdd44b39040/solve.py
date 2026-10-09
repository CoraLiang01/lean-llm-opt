import gurobipy as gp
import pandas as pd
import numpy as np
import re
item_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv'
cutting_patterns_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv'
item_demand_df = pd.read_csv(item_demand_path, sep=',', dtype=str, keep_default_na=False)
item_demand_df['Item'] = item_demand_df['Item'].str.strip()
item_demand_df['Demand'] = item_demand_df['Demand'].astype(int)
items = list(item_demand_df['Item'])
demand_dict = dict(zip(item_demand_df['Item'], item_demand_df['Demand']))
cutting_patterns_df = pd.read_csv(cutting_patterns_path, sep=',', dtype=str, keep_default_na=False)
cutting_patterns_df['Pattern'] = cutting_patterns_df['Pattern'].str.strip()
patterns = list(cutting_patterns_df['Pattern'])
for item in items:
    if item not in cutting_patterns_df.columns:
        raise KeyError(f"Item '{item}' not found as a column in cutting_patterns.csv")
pattern_item_matrix = {}
for (p_idx, row) in cutting_patterns_df.iterrows():
    p = row['Pattern']
    pattern_item_matrix[p] = {}
    for i in items:
        val = row[i]
        try:
            pattern_item_matrix[p][i] = int(val)
        except Exception:
            raise ValueError(f"Invalid integer value for pattern '{p}', item '{i}': '{val}'")
m = gp.Model('CuttingStockPatternSelection')
y_vars = m.addVars(patterns, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((y_vars[p] for p in patterns)), gp.GRB.MINIMIZE)
for i in items:
    m.addConstr(gp.quicksum((pattern_item_matrix[p][i] * y_vars[p] for p in patterns)) >= demand_dict[i], name=f'demand_{i}')
m.optimize()