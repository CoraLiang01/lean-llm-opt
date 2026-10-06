import gurobipy as gp
import pandas as pd
import numpy as np
item_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant18/inputs/item_demand.csv'
item_demand_df = pd.read_csv(item_demand_path, sep=',')
cutting_patterns_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant18/inputs/cutting_patterns.csv'
cutting_patterns_df = pd.read_csv(cutting_patterns_path, sep=',')
items = item_demand_df['Item'].astype(str).tolist()
patterns = cutting_patterns_df['Pattern'].astype(str).tolist()
pattern_item_cols = [col for col in cutting_patterns_df.columns if col != 'Pattern']
if set(items) != set(pattern_item_cols):
    raise ValueError(f'Mismatch between items in item_demand.csv ({items}) and columns in cutting_patterns.csv ({pattern_item_cols})')
demand = dict(zip(item_demand_df['Item'].astype(str), item_demand_df['Demand'].astype(int)))
produced = {}
for idx, row in cutting_patterns_df.iterrows():
    p = str(row['Pattern'])
    produced[p] = {}
    for i in items:
        produced[p][i] = int(row[i])

def solve_cutting_stock(items, patterns, demand, produced):
    m = gp.Model('CuttingStock_MinRolls')
    y = m.addVars(patterns, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((y[p] for p in patterns)), gp.GRB.MINIMIZE)
    for i in items:
        m.addConstr(gp.quicksum((produced[p][i] * y[p] for p in patterns)) >= demand[i], name=f'demand_{i}')
    m.optimize()
    return m
m = solve_cutting_stock(items, patterns, demand, produced)