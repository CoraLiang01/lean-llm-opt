import gurobipy as gp
import pandas as pd
import numpy as np
item_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant9/inputs/item_demand.csv'
cutting_patterns_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant9/inputs/cutting_patterns.csv'
item_demand_df = pd.read_csv(item_demand_path, sep=',')
item_demand_df['Item'] = item_demand_df['Item'].astype(str).str.strip()
items = item_demand_df['Item'].tolist()
demand = dict(zip(item_demand_df['Item'], item_demand_df['Demand']))
patterns_df = pd.read_csv(cutting_patterns_path, sep=',')
patterns_df['Pattern'] = patterns_df['Pattern'].astype(str).str.strip()
patterns = patterns_df['Pattern'].tolist()
missing_items = [i for i in items if i not in patterns_df.columns]
if missing_items:
    raise ValueError(f'Missing item columns in cutting_patterns.csv: {missing_items}')
pattern_item_qty = {p: {i: int(patterns_df.loc[patterns_df['Pattern'] == p, i].values[0]) for i in items} for p in patterns}
m = gp.Model('CuttingStockPatternSelection')
y = m.addVars(patterns, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((y[p] for p in patterns)), gp.GRB.MINIMIZE)
for i in items:
    m.addConstr(gp.quicksum((pattern_item_qty[p][i] * y[p] for p in patterns)) >= demand[i], name=f'demand_{i}')
m.optimize()