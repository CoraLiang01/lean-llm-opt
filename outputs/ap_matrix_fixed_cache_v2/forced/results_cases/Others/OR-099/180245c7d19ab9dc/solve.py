import gurobipy as gp
import pandas as pd
import numpy as np
import re
warehouses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/PotentialWarehouses_Costs.csv', dtype=str, keep_default_na=False)
stores_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/Stores_Demands.csv', dtype=str, keep_default_na=False)
transport_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/TransportationCost.csv', dtype=str, keep_default_na=False)
warehouses_df['Warehouse (i)'] = warehouses_df['Warehouse (i)'].str.strip()
warehouse_ids = warehouses_df['Warehouse (i)'].astype(int).tolist()
stores_df['Store (j)'] = stores_df['Store (j)'].str.strip()
store_ids = stores_df['Store (j)'].astype(int).tolist()
opening_cost = warehouses_df.set_index('Warehouse (i)')['Opening Cost (fi)'].astype(int).to_dict()
capacity = warehouses_df.set_index('Warehouse (i)')['Capacity (units)'].astype(int).to_dict()
demand = stores_df.set_index('Store (j)')['Demand (units, dj)'].astype(int).to_dict()
warehouse_id_to_label = {wid: f'W{wid}' for wid in warehouse_ids}
store_id_to_label = {sid: f'W{sid}' for sid in store_ids}
for wid in warehouse_ids:
    label = warehouse_id_to_label[wid]
    if label not in transport_df.columns:
        raise KeyError(f"TransportationCost.csv missing column for warehouse {wid} ('{label}')")
for sid in store_ids:
    label = store_id_to_label[sid]
    if not (transport_df['Unnamed: 0'].str.strip() == label).any():
        raise KeyError(f"TransportationCost.csv missing row for store {sid} ('{label}')")
c_ij = {}
for wid in warehouse_ids:
    wlabel = warehouse_id_to_label[wid]
    for sid in store_ids:
        slabel = store_id_to_label[sid]
        row = transport_df[transport_df['Unnamed: 0'].str.strip() == slabel]
        if row.empty:
            raise KeyError(f"TransportationCost.csv missing row for store {sid} ('{slabel}')")
        cost_str = row[wlabel].values[0]
        try:
            cost = int(cost_str)
        except Exception:
            raise ValueError(f"Invalid transportation cost for warehouse {wid}, store {sid}: '{cost_str}'")
        c_ij[wid, sid] = cost
m = gp.Model('CapacitatedFacilityLocation')
y_vars = m.addVars(warehouse_ids, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(warehouse_ids, store_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((opening_cost[str(i)] * y_vars[i] for i in warehouse_ids)) + gp.quicksum((c_ij[i, j] * x_vars[i, j] for i in warehouse_ids for j in store_ids)), gp.GRB.MINIMIZE)
for j in store_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in warehouse_ids)) == demand[str(j)], name=f'demand_{j}')
for i in warehouse_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in store_ids)) <= capacity[str(i)] * y_vars[i], name=f'capacity_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_cost = m.objVal
    print(f'Optimal total value/cost: {total_cost:.2f}')
    print('--- Warehouses Opened ---')
    for i in warehouse_ids:
        if y_vars[i].X > 0.5:
            print(f'  Warehouse {i}: OPEN (capacity {capacity[str(i)]}, opening cost {opening_cost[str(i)]})')
    print('--- Shipments ---')
    for i in warehouse_ids:
        for j in store_ids:
            shipped = x_vars[i, j].X
            if shipped > 1e-06:
                print(f'  Warehouse {i} -> Store {j}: {shipped:.2f} units (cost per unit: {c_ij[i, j]})')
else:
    print(f'No optimal solution found. Status: {m.status}')