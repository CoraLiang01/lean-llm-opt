import gurobipy as gp
import pandas as pd
import numpy as np
item_demand_path = 'benchmark_dataset/Variant18/inputs/item_demand.csv'
item_demand_df = pd.read_csv(item_demand_path, sep=',', dtype=str, keep_default_na=False)
item_demand_df['Item'] = item_demand_df['Item'].str.strip()
items = item_demand_df['Item'].tolist()
if not all((item in item_demand_df.columns or item in items for item in items)):
    raise ValueError('Some item IDs are missing in item_demand.csv')
try:
    item_demand_df['Demand'] = item_demand_df['Demand'].astype(int)
except Exception as e:
    raise ValueError('Demand column in item_demand.csv contains non-integer values') from e
item_demand = dict(zip(item_demand_df['Item'], item_demand_df['Demand']))
cutting_patterns_path = 'benchmark_dataset/Variant18/inputs/cutting_patterns.csv'
cutting_patterns_df = pd.read_csv(cutting_patterns_path, sep=',', dtype=str, keep_default_na=False)
cutting_patterns_df['Pattern'] = cutting_patterns_df['Pattern'].str.strip()
patterns = cutting_patterns_df['Pattern'].tolist()
for item in items:
    if item not in cutting_patterns_df.columns:
        raise ValueError(f"Item '{item}' from item_demand.csv is missing as a column in cutting_patterns.csv")
pattern_item_matrix = {}
for (p_idx, row) in cutting_patterns_df.iterrows():
    p = row['Pattern']
    pattern_item_matrix[p] = {}
    for i in items:
        try:
            pattern_item_matrix[p][i] = int(row[i])
        except Exception as e:
            raise ValueError(f"Pattern '{p}', Item '{i}' in cutting_patterns.csv is not an integer") from e
m = gp.Model('CuttingStock_MinRolls')
y_vars = m.addVars(patterns, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((y_vars[p] for p in patterns)), gp.GRB.MINIMIZE)
for i in items:
    m.addConstr(gp.quicksum((pattern_item_matrix[p][i] * y_vars[p] for p in patterns)) >= item_demand[i], name=f'demand_{i}')
m.optimize()