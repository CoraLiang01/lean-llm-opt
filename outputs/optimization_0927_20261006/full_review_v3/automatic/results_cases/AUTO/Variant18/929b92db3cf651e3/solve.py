import gurobipy as gp
import pandas as pd
import numpy as np
import re
item_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant18/inputs/item_demand.csv'
cutting_patterns_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant18/inputs/cutting_patterns.csv'
item_demand_df = pd.read_csv(item_demand_path, sep=',', dtype=str, keep_default_na=False)
item_demand_df['Item'] = item_demand_df['Item'].str.strip()
items = item_demand_df['Item'].tolist()
if item_demand_df['Demand'].isnull().any():
    raise ValueError('Missing demand values in item_demand.csv')
item_demand_df['Demand'] = item_demand_df['Demand'].astype(int)
demand_dict = dict(zip(item_demand_df['Item'], item_demand_df['Demand']))
cutting_patterns_df = pd.read_csv(cutting_patterns_path, sep=',', dtype=str, keep_default_na=False)
cutting_patterns_df['Pattern'] = cutting_patterns_df['Pattern'].str.strip()
patterns = cutting_patterns_df['Pattern'].tolist()
for i in items:
    if i not in cutting_patterns_df.columns:
        raise KeyError(f"Item '{i}' from item_demand.csv not found as a column in cutting_patterns.csv")
pattern_item_coeff = {}
for p in patterns:
    row = cutting_patterns_df.loc[cutting_patterns_df['Pattern'] == p]
    if row.empty:
        raise KeyError(f"Pattern '{p}' not found in cutting_patterns.csv")
    for i in items:
        val = row.iloc[0][i]
        try:
            coeff = int(val)
        except Exception:
            raise ValueError(f"Invalid coefficient for pattern '{p}', item '{i}': {val}")
        pattern_item_coeff[p, i] = coeff
m = gp.Model('CuttingStock_MinRolls')
y_vars = m.addVars(patterns, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((y_vars[p] for p in patterns)), gp.GRB.MINIMIZE)
for i in items:
    m.addConstr(gp.quicksum((pattern_item_coeff[p, i] * y_vars[p] for p in patterns)) >= demand_dict[i], name=f'demand_{i}')
m.optimize()