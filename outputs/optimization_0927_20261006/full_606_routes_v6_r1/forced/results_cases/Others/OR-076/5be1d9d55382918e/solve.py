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
warehouse_ids_set = set(warehouse_ids)
customer_ids = demand_df['Customer ID'].astype(str).str.strip()
customer_ids_set = set(customer_ids)
cost_customer_cols = [col for col in cost_df.columns if col != 'Warehouse ID']
cost_customer_cols_norm = [col.strip() for col in cost_customer_cols]
customer_ids_norm = [cid.strip() for cid in customer_ids]
if set(cost_customer_cols_norm) != set(customer_ids_norm):
    raise ValueError('Mismatch between cost.csv columns and demand.csv Customer IDs.')
col_to_customer = {col.strip(): col.strip() for col in cost_customer_cols}
fixed_cost = {}
capacity = {}
for (idx, row) in warehouse_df.iterrows():
    wid = row['Warehouse ID'].strip()
    if wid not in warehouse_ids_set:
        raise ValueError(f'Warehouse ID {wid} in warehouse.csv not found in cost.csv.')
    try:
        fixed_cost[wid] = float(row['Fixed_Cost'])
        capacity[wid] = float(row['Capacity'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value in warehouse.csv for warehouse {wid}: {e}')
demand = {}
for (idx, row) in demand_df.iterrows():
    cid = row['Customer ID'].strip()
    if cid not in customer_ids_set:
        raise ValueError(f'Customer ID {cid} in demand.csv not found in cost.csv columns.')
    try:
        demand[cid] = float(row['Demand'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value in demand.csv for customer {cid}: {e}')
cost = {}
for (idx, row) in cost_df.iterrows():
    wid = row['Warehouse ID'].strip()
    if wid not in warehouse_ids_set:
        raise ValueError(f'Warehouse ID {wid} in cost.csv not found in warehouse.csv.')
    for col in cost_customer_cols:
        cid = col_to_customer[col]
        if cid not in customer_ids_set:
            raise ValueError(f'Customer column {col} in cost.csv does not match any customer in demand.csv.')
        try:
            cost[wid, cid] = float(row[col])
        except Exception as e:
            raise ValueError(f'Invalid numeric value in cost.csv for warehouse {wid}, customer {cid}: {e}')
m = gp.Model('CapacitatedFacilityLocation')
y_vars = m.addVars(warehouse_ids, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(warehouse_ids, customer_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y_vars[i] for i in warehouse_ids)) + gp.quicksum((cost[i, j] * x_vars[i, j] for i in warehouse_ids for j in customer_ids)), gp.GRB.MINIMIZE)
for j in customer_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in warehouse_ids)) == demand[j], name=f'demand_{j}')
for i in warehouse_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in customer_ids)) <= capacity[i] * y_vars[i], name=f'capacity_{i}')
m.optimize()