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
warehouse_id_to_label = {wid: f'W{wid}' for wid in warehouse_ids}
store_id_to_label = {sid: f'W{sid}' for sid in store_ids}
row_labels = set(transport_df['Unnamed: 0'].astype(str).str.strip())
col_labels = set(transport_df.columns[1:])
for wid in warehouse_ids:
    label = warehouse_id_to_label[wid]
    if label not in row_labels:
        raise ValueError(f"Warehouse label '{label}' not found in TransportationCost.csv rows.")
    if label not in col_labels:
        raise ValueError(f"Warehouse label '{label}' not found in TransportationCost.csv columns.")
for sid in store_ids:
    label = store_id_to_label[sid]
    if label not in col_labels:
        raise ValueError(f"Store label '{label}' not found in TransportationCost.csv columns.")
c_ij = {}
for wid in warehouse_ids:
    row_label = warehouse_id_to_label[wid]
    row = transport_df[transport_df['Unnamed: 0'].astype(str).str.strip() == row_label]
    if row.empty:
        raise ValueError(f"Row for warehouse '{row_label}' not found in TransportationCost.csv.")
    row = row.iloc[0]
    for sid in store_ids:
        col_label = store_id_to_label[sid]
        cost = float(row[col_label])
        c_ij[wid, sid] = cost
m = gp.Model('CapacitatedFacilityLocation')
y = m.addVars(warehouse_ids, vtype=gp.GRB.BINARY, name='')
x = m.addVars(warehouse_ids, store_ids, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((opening_cost[i] * y[i] for i in warehouse_ids)) + gp.quicksum((c_ij[i, j] * x[i, j] for i in warehouse_ids for j in store_ids)), gp.GRB.MINIMIZE)
for j in store_ids:
    m.addConstr(gp.quicksum((x[i, j] for i in warehouse_ids)) == demand[j], name=f'demand_{j}')
for i in warehouse_ids:
    m.addConstr(gp.quicksum((x[i, j] for j in store_ids)) <= capacity[i] * y[i], name=f'capacity_{i}')
for i in warehouse_ids:
    for j in store_ids:
        m.addConstr(x[i, j] <= demand[j] * y[i], name=f'assign_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Warehouses Opened ---')
    for i in warehouse_ids:
        if y[i].X > 0.5:
            print(f'  Warehouse {i}: OPEN (capacity {int(capacity[i])}, opening cost {int(opening_cost[i])})')
    print('--- Store Assignments ---')
    for j in store_ids:
        print(f'Store {j} (demand {int(demand[j])}):')
        for i in warehouse_ids:
            if x[i, j].X > 1e-06:
                print(f'  Supplied {x[i, j].X:.2f} units from Warehouse {i} (cost per unit: {c_ij[i, j]})')
else:
    print(f'No optimal solution found. Status: {m.status}')