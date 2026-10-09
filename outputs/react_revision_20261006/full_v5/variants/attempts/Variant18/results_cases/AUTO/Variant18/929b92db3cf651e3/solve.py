import gurobipy as gp
import pandas as pd
import numpy as np
item_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant18/inputs/item_demand.csv'
cutting_patterns_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant18/inputs/cutting_patterns.csv'
item_demand_df = pd.read_csv(item_demand_path, sep=',')
item_demand_df['Item'] = item_demand_df['Item'].astype(str).str.strip()
items = item_demand_df['Item'].tolist()
demands = dict(zip(item_demand_df['Item'], item_demand_df['Demand']))
patterns_df = pd.read_csv(cutting_patterns_path, sep=',')
patterns_df['Pattern'] = patterns_df['Pattern'].astype(str).str.strip()
patterns = patterns_df['Pattern'].tolist()
pattern_item_cols = [col for col in patterns_df.columns if col != 'Pattern']
missing_items = set(items) - set(pattern_item_cols)
if missing_items:
    raise ValueError(f'Missing item columns in cutting_patterns.csv: {missing_items}')
pieces = {}
for i in items:
    pieces[i] = {}
    for p in patterns:
        val = patterns_df.loc[patterns_df['Pattern'] == p, i]
        if val.empty:
            raise ValueError(f"Pattern '{p}' missing item '{i}' in cutting_patterns.csv")
        pieces[i][p] = int(val.values[0])
m = gp.Model('CuttingStock_MinRolls')
y = m.addVars(patterns, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((y[p] for p in patterns)), gp.GRB.MINIMIZE)
for i in items:
    m.addConstr(gp.quicksum((pieces[i][p] * y[p] for p in patterns)) >= demands[i], name=f'demand_{i}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for p in patterns:
        print(f'{y[p].VarName} {y[p].X}')
else:
    print(f'Solver status: {m.status}')