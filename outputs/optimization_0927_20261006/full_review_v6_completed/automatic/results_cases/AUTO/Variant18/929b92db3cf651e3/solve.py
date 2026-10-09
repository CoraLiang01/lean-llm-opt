import gurobipy as gp
import pandas as pd
import numpy as np
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
        raise KeyError(f"cutting_patterns.csv missing required item column '{item}'.")
pieces_per_pattern = {}
for (_, row) in cutting_patterns_df.iterrows():
    p = row['Pattern']
    for i in items:
        try:
            pieces_per_pattern[p, i] = int(row[i])
        except ValueError:
            raise ValueError(f"Invalid integer value for pattern '{p}', item '{i}' in cutting_patterns.csv.")
m = gp.Model('CuttingStock_MinRolls')
y_vars = m.addVars(patterns, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((y_vars[p] for p in patterns)), gp.GRB.MINIMIZE)
for i in items:
    m.addConstr(gp.quicksum((pieces_per_pattern[p, i] * y_vars[p] for p in patterns)) >= demands[i], name=f'demand_{i}')
m.optimize()