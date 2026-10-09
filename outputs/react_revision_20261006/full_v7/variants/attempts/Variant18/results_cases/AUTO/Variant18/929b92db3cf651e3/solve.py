import gurobipy as gp
import pandas as pd
import numpy as np
item_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant18/inputs/item_demand.csv'
cutting_patterns_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant18/inputs/cutting_patterns.csv'
item_demand_df = pd.read_csv(item_demand_path, sep=',', dtype=str, keep_default_na=False)
cutting_patterns_df = pd.read_csv(cutting_patterns_path, sep=',', dtype=str, keep_default_na=False)
item_demand_df['Item'] = item_demand_df['Item'].str.strip()
cutting_patterns_df['Pattern'] = cutting_patterns_df['Pattern'].str.strip()
items = item_demand_df['Item'].unique().tolist()
patterns = cutting_patterns_df['Pattern'].unique().tolist()
for item in items:
    if item not in cutting_patterns_df.columns:
        raise ValueError(f"Item '{item}' from item_demand.csv not found as a column in cutting_patterns.csv.")
demand = {}
for (_, row) in item_demand_df.iterrows():
    item = row['Item']
    try:
        demand[item] = int(row['Demand'])
    except Exception:
        raise ValueError(f"Demand for item '{item}' is not a valid integer: {row['Demand']}")
pattern_item = {}
for (_, row) in cutting_patterns_df.iterrows():
    pattern = row['Pattern']
    for item in items:
        try:
            pattern_item[pattern, item] = int(row[item])
        except Exception:
            raise ValueError(f"Pattern '{pattern}', item '{item}' has invalid value: {row[item]}")
for pattern in patterns:
    for item in items:
        if (pattern, item) not in pattern_item:
            raise ValueError(f"Missing coefficient for pattern '{pattern}', item '{item}'.")

def solve_cutting_stock(items, patterns, demand, pattern_item):
    m = gp.Model('cutting_stock_min_rolls')
    y_vars = m.addVars(patterns, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((y_vars[p] for p in patterns)), gp.GRB.MINIMIZE)
    for item in items:
        m.addConstr(gp.quicksum((pattern_item[p, item] * y_vars[p] for p in patterns)) >= demand[item], name=f'demand_{item}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_cutting_stock(items, patterns, demand, pattern_item)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')