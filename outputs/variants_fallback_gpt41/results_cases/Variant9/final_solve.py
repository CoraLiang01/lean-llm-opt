import gurobipy as gp
import pandas as pd
import numpy as np
item_demand_path = 'benchmark_dataset/Variant9/inputs/item_demand.csv'
cutting_patterns_path = 'benchmark_dataset/Variant9/inputs/cutting_patterns.csv'
item_demand_df = pd.read_csv(item_demand_path, sep=',', dtype=str, keep_default_na=False)
if 'Item' not in item_demand_df.columns or 'Demand' not in item_demand_df.columns:
    raise KeyError("item_demand.csv must contain columns 'Item' and 'Demand'.")
item_demand_df['Item'] = item_demand_df['Item'].str.strip()
item_ids = item_demand_df['Item'].tolist()
item_demand_df['Demand'] = item_demand_df['Demand'].astype(int)
demand_dict = dict(zip(item_demand_df['Item'], item_demand_df['Demand']))
cutting_patterns_df = pd.read_csv(cutting_patterns_path, sep=',', dtype=str, keep_default_na=False)
if 'Pattern' not in cutting_patterns_df.columns:
    raise KeyError("cutting_patterns.csv must contain column 'Pattern'.")
cutting_patterns_df['Pattern'] = cutting_patterns_df['Pattern'].str.strip()
pattern_ids = cutting_patterns_df['Pattern'].tolist()
for item in item_ids:
    if item not in cutting_patterns_df.columns:
        raise KeyError(f"cutting_patterns.csv missing required item column '{item}'.")
pattern_matrix = {}
for p_row in cutting_patterns_df.itertuples(index=False):
    p_id = getattr(p_row, 'Pattern')
    pattern_matrix[p_id] = {}
    for item in item_ids:
        val = getattr(p_row, item)
        try:
            pattern_matrix[p_id][item] = int(val)
        except Exception:
            raise ValueError(f"Invalid integer value for pattern '{p_id}', item '{item}': '{val}'")
m = gp.Model('CuttingStockPatternSelection')
y_vars = m.addVars(pattern_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((y_vars[p] for p in pattern_ids)), gp.GRB.MINIMIZE)
for item in item_ids:
    m.addConstr(gp.quicksum((pattern_matrix[p][item] * y_vars[p] for p in pattern_ids)) >= demand_dict[item], name=f'demand_{item}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total number of standard rolls used: {m.objVal:.0f}')
    print('--- Pattern Usage ---')
    for p in pattern_ids:
        qty = y_vars[p].X
        if qty > 1e-06:
            print(f'Pattern {p}: {qty:.0f} times')
    print('--- Demand Satisfaction ---')
    for item in item_ids:
        produced = sum((pattern_matrix[p][item] * y_vars[p].X for p in pattern_ids))
        print(f'Item {item}: Demand = {demand_dict[item]}, Produced = {int(round(produced))}')
else:
    print(f'No optimal solution found. Status: {m.status}')