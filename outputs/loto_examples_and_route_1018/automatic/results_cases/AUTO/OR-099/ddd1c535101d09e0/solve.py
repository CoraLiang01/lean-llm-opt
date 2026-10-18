import gurobipy as gp
import pandas as pd
import numpy as np
import re
warehouses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/PotentialWarehouses_Costs.csv', dtype=str, keep_default_na=False)
stores_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/Stores_Demands.csv', dtype=str, keep_default_na=False)
transport_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/TransportationCost.csv', dtype=str, keep_default_na=False)
warehouse_ids = warehouses_df['Warehouse (i)'].astype(int).tolist()
opening_cost = warehouses_df.set_index(warehouses_df['Warehouse (i)'].astype(int))['Opening Cost (fi)'].astype(int).to_dict()
capacity = warehouses_df.set_index(warehouses_df['Warehouse (i)'].astype(int))['Capacity (units)'].astype(int).to_dict()
store_ids = stores_df['Store (j)'].astype(int).tolist()
demand = stores_df.set_index(stores_df['Store (j)'].astype(int))['Demand (units, dj)'].astype(int).to_dict()
warehouse_id_to_label = {wid: f'W{wid}' for wid in warehouse_ids}
warehouse_label_to_id = {f'W{wid}': wid for wid in warehouse_ids}
row_labels = transport_df['Unnamed: 0'].tolist()
col_labels = [col for col in transport_df.columns if col != 'Unnamed: 0']
expected_labels = [f'W{wid}' for wid in warehouse_ids]
if set(row_labels) != set(expected_labels) or set(col_labels) != set(expected_labels):
    raise ValueError('Mismatch between warehouse/store IDs and TransportationCost.csv labels.')
transport_cost = {}
for i in warehouse_ids:
    row_label = warehouse_id_to_label[i]
    row = transport_df[transport_df['Unnamed: 0'] == row_label]
    if row.empty:
        raise ValueError(f'Missing row for warehouse {i} in TransportationCost.csv')
    for j in store_ids:
        col_label = warehouse_id_to_label[j]
        try:
            cij = int(row.iloc[0][col_label])
        except Exception as e:
            raise ValueError(f'Missing or invalid transportation cost for warehouse {i} to store {j}: {e}')
        transport_cost[i, j] = cij
m = gp.Model('CapacitatedFacilityLocation')
y_vars = m.addVars(warehouse_ids, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(warehouse_ids, store_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((opening_cost[i] * y_vars[i] for i in warehouse_ids)) + gp.quicksum((transport_cost[i, j] * x_vars[i, j] for i in warehouse_ids for j in store_ids)), gp.GRB.MINIMIZE)
for j in store_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in warehouse_ids)) == demand[j], name=f'demand_{j}')
for i in warehouse_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in store_ids)) <= capacity[i] * y_vars[i], name=f'capacity_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Warehouses Opened ---')
    for i in warehouse_ids:
        if y_vars[i].X > 0.5:
            print(f'Warehouse {i}: OPEN (Opening cost: {opening_cost[i]})')
    print('--- Shipments ---')
    for i in warehouse_ids:
        for j in store_ids:
            if x_vars[i, j].X > 1e-06:
                print(f'Warehouse {i} -> Store {j}: {x_vars[i, j].X:.2f} units (Cost per unit: {transport_cost[i, j]})')
else:
    print(f'No optimal solution found. Status: {m.status}')