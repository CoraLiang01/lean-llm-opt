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
warehouse_ids = warehouse_df['Warehouse ID'].str.strip().tolist()
customer_ids = demand_df['Customer ID'].str.strip().tolist()
fixed_cost = {}
capacity = {}
for idx, row in warehouse_df.iterrows():
    wid = row['Warehouse ID'].strip()
    fixed_cost[wid] = int(row['Fixed_Cost'])
    capacity[wid] = int(row['Capacity'])
demand = {}
for idx, row in demand_df.iterrows():
    cid = row['Customer ID'].strip()
    demand[cid] = int(row['Demand'])
cost_per_unit = {}
cost_columns = [col for col in cost_df.columns if col != 'Warehouse ID']
expected_cost_columns = ['C' + str(int(re.sub('\\D', '', cid))) for cid in customer_ids]
if set(expected_cost_columns) != set(cost_columns):
    raise ValueError('Mismatch between customer IDs and cost.csv columns.')
for idx, row in cost_df.iterrows():
    wid = row['Warehouse ID'].strip()
    cost_per_unit[wid] = {}
    for cid in customer_ids:
        ccol = 'C' + str(int(re.sub('\\D', '', cid)))
        cost_per_unit[wid][cid] = int(row[ccol])
if set(warehouse_ids) != set(cost_df['Warehouse ID'].str.strip()):
    raise ValueError('Mismatch between warehouse IDs in warehouse.csv and cost.csv.')
if set(customer_ids) != set([col for col in cost_columns]):
    raise ValueError('Mismatch between customer IDs in demand.csv and cost.csv columns.')
m = gp.Model('CapacitatedFacilityLocation')
y_vars = m.addVars(warehouse_ids, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(warehouse_ids, customer_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y_vars[i] for i in warehouse_ids)) + gp.quicksum((cost_per_unit[i][j] * x_vars[i, j] for i in warehouse_ids for j in customer_ids)), gp.GRB.MINIMIZE)
for j in customer_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in warehouse_ids)) == demand[j], name=f'demand_{j}')
for i in warehouse_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in customer_ids)) <= capacity[i] * y_vars[i], name=f'capacity_{i}')
m.optimize()