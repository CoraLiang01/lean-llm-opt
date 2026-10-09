import gurobipy as gp
import pandas as pd
import numpy as np
import re
item_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant18/inputs/item_demand.csv'
item_demand_df = pd.read_csv(item_demand_path, sep=',', dtype=str, keep_default_na=False)
item_demand_df['Item_norm'] = item_demand_df['Item'].str.strip().str.casefold()
item_demand_df['Demand'] = item_demand_df['Demand'].astype(int)
items = list(item_demand_df['Item'])
items_norm = list(item_demand_df['Item_norm'])
item_id_map = dict(zip(items_norm, items))
demand_dict = dict(zip(items, item_demand_df['Demand']))
cutting_patterns_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant18/inputs/cutting_patterns.csv'
patterns_df = pd.read_csv(cutting_patterns_path, sep=',', dtype=str, keep_default_na=False)
patterns_df['Pattern_norm'] = patterns_df['Pattern'].str.strip().str.casefold()
patterns = list(patterns_df['Pattern'])
patterns_norm = list(patterns_df['Pattern_norm'])
pattern_id_map = dict(zip(patterns_norm, patterns))
pattern_item_cols = [col for col in patterns_df.columns if col in items]
if set(items) != set(pattern_item_cols):
    missing = set(items) - set(pattern_item_cols)
    extra = set(pattern_item_cols) - set(items)
    raise ValueError(f'Mismatch between items in item_demand.csv and columns in cutting_patterns.csv. Missing: {missing}, Extra: {extra}')
pieces = {}
for (p_idx, p_row) in patterns_df.iterrows():
    p = p_row['Pattern']
    pieces[p] = {}
    for i in items:
        val = int(p_row[i])
        pieces[p][i] = val
m = gp.Model('CuttingStock_MinRolls')
y_vars = m.addVars(patterns, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((y_vars[p] for p in patterns)), gp.GRB.MINIMIZE)
for i in items:
    m.addConstr(gp.quicksum((pieces[p][i] * y_vars[p] for p in patterns)) >= demand_dict[i], name=f'demand_{i}')
m.optimize()