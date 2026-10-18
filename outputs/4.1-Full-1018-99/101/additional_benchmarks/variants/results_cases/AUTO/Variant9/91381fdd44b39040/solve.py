import gurobipy as gp
import pandas as pd
import numpy as np
item_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv'
cutting_patterns_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv'
item_demand_df = pd.read_csv(item_demand_path, sep=',', dtype=str, keep_default_na=False)
item_demand_df['Item'] = item_demand_df['Item'].str.strip()
item_demand_df['Demand'] = item_demand_df['Demand'].astype(int)
cutting_patterns_df = pd.read_csv(cutting_patterns_path, sep=',', dtype=str, keep_default_na=False)
cutting_patterns_df['Pattern'] = cutting_patterns_df['Pattern'].str.strip()
item_columns = ['A', 'B', 'C', 'D']
for col in item_columns:
    cutting_patterns_df[col] = cutting_patterns_df[col].astype(int)
patterns = list(cutting_patterns_df['Pattern'])
items = list(item_demand_df['Item'])
missing_items = [i for i in items if i not in cutting_patterns_df.columns]
if missing_items:
    raise ValueError(f'Missing item columns in cutting_patterns.csv: {missing_items}')
demand = dict(zip(item_demand_df['Item'], item_demand_df['Demand']))
pattern_item_units = {}
for _, row in cutting_patterns_df.iterrows():
    p = row['Pattern']
    for i in items:
        pattern_item_units[p, i] = int(row[i])
m = gp.Model('CuttingStockPatternSelection')
y_vars = m.addVars(patterns, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((y_vars[p] for p in patterns)), gp.GRB.MINIMIZE)
for i in items:
    m.addConstr(gp.quicksum((pattern_item_units[p, i] * y_vars[p] for p in patterns)) >= demand[i], name=f'demand_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total number of standard rolls used: {m.objVal:.0f}')
    print('--- Pattern Usage ---')
    for p in patterns:
        val = y_vars[p].X
        if val > 1e-06:
            print(f'Pattern {p}: {val:.0f} times')
    print('--- Demand Satisfaction ---')
    for i in items:
        produced = sum((pattern_item_units[p, i] * y_vars[p].X for p in patterns))
        print(f'Item {i}: Demand = {demand[i]}, Produced = {int(round(produced))}')
else:
    print(f'No optimal solution found. Status: {m.status}')