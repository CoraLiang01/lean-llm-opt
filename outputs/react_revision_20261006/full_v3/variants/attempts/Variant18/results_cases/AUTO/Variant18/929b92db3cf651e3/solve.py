import gurobipy as gp
import pandas as pd
import numpy as np
item_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant18/inputs/item_demand.csv'
cutting_patterns_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant18/inputs/cutting_patterns.csv'
item_demand_df = pd.read_csv(item_demand_path, sep=',')
if item_demand_df['Item'].duplicated().any():
    raise ValueError('Duplicate Item identifiers found in item_demand.csv')
items = item_demand_df['Item'].astype(str).tolist()
demands = dict(zip(item_demand_df['Item'].astype(str), item_demand_df['Demand'].astype(int)))
cutting_patterns_df = pd.read_csv(cutting_patterns_path, sep=',')
if cutting_patterns_df['Pattern'].duplicated().any():
    raise ValueError('Duplicate Pattern identifiers found in cutting_patterns.csv')
patterns = cutting_patterns_df['Pattern'].astype(str).tolist()
missing_items = [i for i in items if i not in cutting_patterns_df.columns]
if missing_items:
    raise ValueError(f'Items {missing_items} from item_demand.csv not found as columns in cutting_patterns.csv')
pieces = {}
for p in patterns:
    row = cutting_patterns_df.loc[cutting_patterns_df['Pattern'].astype(str) == p]
    if row.empty:
        raise ValueError(f'Pattern {p} not found in cutting_patterns.csv')
    for i in items:
        val = int(row.iloc[0][i])
        pieces[p, i] = val
m = gp.Model('cutting_stock')
m.setParam('MIPGap', 0.0001)
y = m.addVars(patterns, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((y[p] for p in patterns)), gp.GRB.MINIMIZE)
for i in items:
    m.addConstr(gp.quicksum((pieces[p, i] * y[p] for p in patterns)) >= demands[i], name=f'demand_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for p in patterns:
        print(f'{y[p].VarName} {y[p].X}')
else:
    print(f'Solver status: {m.status}')