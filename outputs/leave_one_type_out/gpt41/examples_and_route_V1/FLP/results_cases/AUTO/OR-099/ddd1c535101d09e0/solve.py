import gurobipy as gp
import pandas as pd
import numpy as np
import re
warehouses_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/PotentialWarehouses_Costs.csv'
stores_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/Stores_Demands.csv'
transport_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/TransportationCost.csv'
warehouses_df = pd.read_csv(warehouses_path, sep=',')
stores_df = pd.read_csv(stores_path, sep=',')
transport_df = pd.read_csv(transport_path, sep=',')
warehouse_ids = warehouses_df['Warehouse (i)'].astype(int).tolist()
opening_cost = warehouses_df.set_index('Warehouse (i)')['Opening Cost (fi)'].astype(float).to_dict()
capacity = warehouses_df.set_index('Warehouse (i)')['Capacity (units)'].astype(float).to_dict()
store_ids = stores_df['Store (j)'].astype(int).tolist()
demand = stores_df.set_index('Store (j)')['Demand (units, dj)'].astype(float).to_dict()

def wid_to_label(wid):
    return f'W{int(wid)}'
warehouse_labels = [wid_to_label(wid) for wid in warehouse_ids]
store_labels = [wid_to_label(sid) for sid in store_ids]
if not all((lbl in transport_df.columns for lbl in warehouse_labels)):
    raise ValueError('Not all warehouse labels found in TransportationCost.csv columns.')
if not all((lbl in transport_df['Unnamed: 0'].values for lbl in store_labels)):
    raise ValueError('Not all store labels found in TransportationCost.csv row labels.')
c_ij = {}
for j, store_id in enumerate(store_ids):
    row_label = wid_to_label(store_id)
    row = transport_df[transport_df['Unnamed: 0'] == row_label]
    if row.empty:
        raise ValueError(f'Store label {row_label} not found in TransportationCost.csv rows.')
    for i, warehouse_id in enumerate(warehouse_ids):
        col_label = wid_to_label(warehouse_id)
        if col_label not in transport_df.columns:
            raise ValueError(f'Warehouse label {col_label} not found in TransportationCost.csv columns.')
        cost = float(row.iloc[0][col_label])
        c_ij[warehouse_id, store_id] = cost
m = gp.Model('CapacitatedFacilityLocation')
y = m.addVars(warehouse_ids, vtype=gp.GRB.BINARY, name='')
x = m.addVars(warehouse_ids, store_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((opening_cost[i] * y[i] for i in warehouse_ids)) + gp.quicksum((c_ij[i, j] * x[i, j] for i in warehouse_ids for j in store_ids)), gp.GRB.MINIMIZE)
for j in store_ids:
    m.addConstr(gp.quicksum((x[i, j] for i in warehouse_ids)) == demand[j], name=f'demand_{j}')
for i in warehouse_ids:
    m.addConstr(gp.quicksum((x[i, j] for j in store_ids)) <= capacity[i] * y[i], name=f'capacity_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Warehouses Opened ---')
    for i in warehouse_ids:
        if y[i].X > 0.5:
            print(f'Warehouse {i}: OPEN (Opening cost: {opening_cost[i]:.2f}, Capacity: {capacity[i]:.0f})')
    print('--- Store Assignments ---')
    for j in store_ids:
        print(f'Store {j} (Demand: {demand[j]:.0f}):')
        for i in warehouse_ids:
            if x[i, j].X > 1e-06:
                print(f'  Supplied {x[i, j].X:.2f} units from Warehouse {i} (Cost per unit: {c_ij[i, j]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')