import gurobipy as gp
import pandas as pd
import numpy as np
import re
warehouses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/PotentialWarehouses_Costs.csv', dtype=str, keep_default_na=False)
stores_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/Stores_Demands.csv', dtype=str, keep_default_na=False)
transport_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/TransportationCost.csv', dtype=str, keep_default_na=False)
warehouses_df['Warehouse (i)'] = warehouses_df['Warehouse (i)'].astype(int)
warehouse_ids = warehouses_df['Warehouse (i)'].tolist()
stores_df['Store (j)'] = stores_df['Store (j)'].astype(int)
store_ids = stores_df['Store (j)'].tolist()
opening_cost = dict(zip(warehouses_df['Warehouse (i)'], warehouses_df['Opening Cost (fi)'].astype(float)))
capacity = dict(zip(warehouses_df['Warehouse (i)'], warehouses_df['Capacity (units)'].astype(float)))
demand = dict(zip(stores_df['Store (j)'], stores_df['Demand (units, dj)'].astype(float)))
warehouse_id_to_label = {wid: f'W{wid}' for wid in warehouse_ids}
store_id_to_label = {sid: f'W{sid}' for sid in store_ids}
transport_row_labels = transport_df['Unnamed: 0'].tolist()
transport_col_labels = [col for col in transport_df.columns if col != 'Unnamed: 0']
for wid in warehouse_ids:
    if warehouse_id_to_label[wid] not in transport_row_labels:
        raise ValueError(f'Warehouse label {warehouse_id_to_label[wid]} not found in TransportationCost.csv rows')
for sid in store_ids:
    if store_id_to_label[sid] not in transport_col_labels:
        raise ValueError(f'Store label {store_id_to_label[sid]} not found in TransportationCost.csv columns')
c_ij = {}
for wid in warehouse_ids:
    row_label = warehouse_id_to_label[wid]
    row = transport_df[transport_df['Unnamed: 0'] == row_label]
    if row.empty:
        raise ValueError(f'Row for warehouse {row_label} not found in TransportationCost.csv')
    row = row.iloc[0]
    for sid in store_ids:
        col_label = store_id_to_label[sid]
        cost_val = float(row[col_label])
        c_ij[wid, sid] = cost_val
m = gp.Model('CapacitatedFacilityLocation')
y_vars = m.addVars(warehouse_ids, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(warehouse_ids, store_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((opening_cost[i] * y_vars[i] for i in warehouse_ids)) + gp.quicksum((c_ij[i, j] * x_vars[i, j] for i in warehouse_ids for j in store_ids)), gp.GRB.MINIMIZE)
for j in store_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in warehouse_ids)) == demand[j], name=f'demand_{j}')
for i in warehouse_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in store_ids)) <= capacity[i] * y_vars[i], name=f'capacity_{i}')
m.optimize()