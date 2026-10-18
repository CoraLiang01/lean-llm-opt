import gurobipy as gp
import pandas as pd
import numpy as np
import re
item_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant18/inputs/item_demand.csv'
item_demand_df = pd.read_csv(item_demand_path, sep=',', dtype=str, keep_default_na=False)
item_demand_df['Item'] = item_demand_df['Item'].str.strip()
items = item_demand_df['Item'].tolist()
if not all((item in item_demand_df.columns or item in items for item in items)):
    raise ValueError('Mismatch between item identifiers in demand file.')
try:
    item_demand_df['Demand'] = item_demand_df['Demand'].astype(int)
except Exception as e:
    raise ValueError(f'Failed to convert Demand column to int: {e}')
demand_dict = dict(zip(item_demand_df['Item'], item_demand_df['Demand']))
cutting_patterns_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant18/inputs/cutting_patterns.csv'
cutting_patterns_df = pd.read_csv(cutting_patterns_path, sep=',', dtype=str, keep_default_na=False)
cutting_patterns_df['Pattern'] = cutting_patterns_df['Pattern'].str.strip()
patterns = cutting_patterns_df['Pattern'].tolist()
for item in items:
    if item not in cutting_patterns_df.columns:
        raise KeyError(f"Item '{item}' from demand file not found as a column in cutting_patterns.csv.")
pieces = {}
for item in items:
    try:
        cutting_patterns_df[item] = cutting_patterns_df[item].astype(int)
    except Exception as e:
        raise ValueError(f"Failed to convert pattern column '{item}' to int: {e}")
for item in items:
    pieces[item] = {}
    for idx, row in cutting_patterns_df.iterrows():
        pattern = row['Pattern']
        pieces[item][pattern] = int(row[item])
m = gp.Model('CuttingStock_MinRolls')
y_vars = m.addVars(patterns, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((y_vars[p] for p in patterns)), gp.GRB.MINIMIZE)
for item in items:
    m.addConstr(gp.quicksum((pieces[item][p] * y_vars[p] for p in patterns)) >= demand_dict[item], name=f'demand_{item}')
m.optimize()