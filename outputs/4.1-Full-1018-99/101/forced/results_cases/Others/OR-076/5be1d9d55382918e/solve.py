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
warehouse_ids = warehouse_df['Warehouse ID'].astype(str).str.strip()
cost_warehouse_ids = cost_df['Warehouse ID'].astype(str).str.strip()
if not set(warehouse_ids) == set(cost_warehouse_ids):
    raise ValueError('Mismatch between warehouse IDs in warehouse.csv and cost.csv')
warehouses = list(warehouse_ids)
customer_ids = demand_df['Customer ID'].astype(str).str.strip()
cost_customer_cols = [col for col in cost_df.columns if re.fullmatch('C\\d+', col.strip())]
if len(cost_customer_cols) != len(customer_ids):
    raise ValueError('Mismatch between number of customers in demand.csv and cost.csv')
customers = list(customer_ids)
fixed_cost = {}
capacity = {}
for _, row in warehouse_df.iterrows():
    wid = str(row['Warehouse ID']).strip()
    fixed_cost[wid] = int(row['Fixed_Cost'])
    capacity[wid] = int(row['Capacity'])
demand = {}
for _, row in demand_df.iterrows():
    cid = str(row['Customer ID']).strip()
    demand[cid] = int(row['Demand'])
transport_cost = {}
for _, row in cost_df.iterrows():
    wid = str(row['Warehouse ID']).strip()
    transport_cost[wid] = {}
    for cidx, cid in enumerate(customers):
        col = cost_customer_cols[cidx]
        transport_cost[wid][cid] = int(row[col])
m = gp.Model('CapacitatedFacilityLocation')
y_vars = m.addVars(warehouses, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(warehouses, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((fixed_cost[w] * y_vars[w] for w in warehouses)) + gp.quicksum((transport_cost[w][c] * x_vars[w, c] for w in warehouses for c in customers)), gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x_vars[w, c] for w in warehouses)) == demand[c], name=f'demand_{c}')
for w in warehouses:
    m.addConstr(gp.quicksum((x_vars[w, c] for c in customers)) <= capacity[w] * y_vars[w], name=f'capacity_{w}')
m.optimize()