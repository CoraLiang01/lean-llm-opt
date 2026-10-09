import gurobipy as gp
import pandas as pd
import numpy as np
import re
item_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant18/inputs/item_demand.csv'
cutting_patterns_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant18/inputs/cutting_patterns.csv'
item_demand_df = pd.read_csv(item_demand_path, sep=',', dtype=str, keep_default_na=False)
if 'Item' not in item_demand_df.columns or 'Demand' not in item_demand_df.columns:
    raise KeyError("item_demand.csv must contain columns 'Item' and 'Demand'.")
item_demand_df['Item'] = item_demand_df['Item'].astype(str).str.strip()
items = item_demand_df['Item'].tolist()
item_demand_df['Demand'] = item_demand_df['Demand'].astype(int)
demand = dict(zip(item_demand_df['Item'], item_demand_df['Demand']))
cutting_patterns_df = pd.read_csv(cutting_patterns_path, sep=',', dtype=str, keep_default_na=False)
if 'Pattern' not in cutting_patterns_df.columns:
    raise KeyError("cutting_patterns.csv must contain column 'Pattern'.")
cutting_patterns_df['Pattern'] = cutting_patterns_df['Pattern'].astype(str).str.strip()
patterns = cutting_patterns_df['Pattern'].tolist()
for item in items:
    if item not in cutting_patterns_df.columns:
        raise KeyError(f"cutting_patterns.csv missing column for item '{item}'.")
a = {}
for (idx, row) in cutting_patterns_df.iterrows():
    p = row['Pattern']
    a[p] = {}
    for i in items:
        val = row[i]
        try:
            a[p][i] = int(val)
        except Exception:
            raise ValueError(f"Invalid integer value for pattern '{p}', item '{i}': '{val}'")
m = gp.Model('CuttingStock_MinRolls')
y_vars = m.addVars(patterns, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((y_vars[p] for p in patterns)), gp.GRB.MINIMIZE)
for i in items:
    m.addConstr(gp.quicksum((a[p][i] * y_vars[p] for p in patterns)) >= demand[i], name=f'demand_{i}')
m.optimize()