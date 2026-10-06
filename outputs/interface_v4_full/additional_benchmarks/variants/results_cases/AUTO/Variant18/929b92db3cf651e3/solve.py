import gurobipy as gp
import pandas as pd
import numpy as np
item_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant18/inputs/item_demand.csv'
item_demand_df = pd.read_csv(item_demand_path, sep=',')
cutting_patterns_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant18/inputs/cutting_patterns.csv'
cutting_patterns_df = pd.read_csv(cutting_patterns_path, sep=',')
item_demand_df['Item'] = item_demand_df['Item'].astype(str).str.strip()
cutting_patterns_df['Pattern'] = cutting_patterns_df['Pattern'].astype(str).str.strip()
items = list(item_demand_df['Item'])
patterns = list(cutting_patterns_df['Pattern'])
demand = dict(zip(item_demand_df['Item'], item_demand_df['Demand']))
produced = {}
for p in patterns:
    row = cutting_patterns_df.loc[cutting_patterns_df['Pattern'] == p]
    if row.empty:
        raise ValueError(f"Pattern '{p}' not found in cutting_patterns.csv")
    for i in items:
        if i not in cutting_patterns_df.columns:
            raise ValueError(f"Item '{i}' not found as a column in cutting_patterns.csv")
        val = int(row.iloc[0][i])
        produced[i, p] = val
m = gp.Model('CuttingStock_MinRolls')
y = m.addVars(patterns, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((y[p] for p in patterns)), gp.GRB.MINIMIZE)
for i in items:
    m.addConstr(gp.quicksum((produced[i, p] * y[p] for p in patterns)) >= demand[i], name=f'demand_{i}')
m.optimize()