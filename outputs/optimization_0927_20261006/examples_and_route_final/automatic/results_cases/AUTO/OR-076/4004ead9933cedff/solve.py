import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/cost.csv'
warehouse_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/warehouse.csv'
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/demand.csv'
cost_df = pd.read_csv(cost_path, dtype=str, keep_default_na=False)
warehouse_df = pd.read_csv(warehouse_path, dtype=str, keep_default_na=False)
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
warehouse_ids = warehouse_df['Warehouse ID'].tolist()
customer_ids = demand_df['Customer ID'].tolist()
cost_warehouse_ids = cost_df['Warehouse ID'].tolist()
if set(warehouse_ids) != set(cost_warehouse_ids):
    raise ValueError('Mismatch between warehouse IDs in warehouse.csv and cost.csv')
cost_customer_cols = [col for col in cost_df.columns if re.fullmatch('C\\d+', col)]
expected_customer_cols = ['C{}'.format(i + 1) for i in range(len(customer_ids))]
if cost_customer_cols != expected_customer_cols:
    raise ValueError('Customer columns in cost.csv do not match expected order from demand.csv')
customer_id_to_col = dict(zip(customer_ids, expected_customer_cols))
col_to_customer_id = dict(zip(expected_customer_cols, customer_ids))
fixed_cost = {}
capacity = {}
for (idx, row) in warehouse_df.iterrows():
    wid = row['Warehouse ID']
    fixed_cost[wid] = int(row['Fixed_Cost'])
    capacity[wid] = int(row['Capacity'])
demand = {}
for (idx, row) in demand_df.iterrows():
    cid = row['Customer ID']
    demand[cid] = int(row['Demand'])
cost = {}
for (idx, row) in cost_df.iterrows():
    wid = row['Warehouse ID']
    cost[wid] = {}
    for ccol in expected_customer_cols:
        cid = col_to_customer_id[ccol]
        cost[wid][cid] = int(row[ccol])
m = gp.Model('CapacitatedFacilityLocation')
y_vars = m.addVars(warehouse_ids, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(warehouse_ids, customer_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y_vars[i] for i in warehouse_ids)) + gp.quicksum((cost[i][j] * x_vars[i, j] for i in warehouse_ids for j in customer_ids)), gp.GRB.MINIMIZE)
for j in customer_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in warehouse_ids)) == demand[j], name=f'demand_{j}')
for i in warehouse_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in customer_ids)) <= capacity[i] * y_vars[i], name=f'capacity_{i}')
m.optimize()