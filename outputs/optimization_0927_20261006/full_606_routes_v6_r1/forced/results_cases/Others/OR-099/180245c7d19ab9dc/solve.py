import gurobipy as gp
import pandas as pd
import numpy as np
import re
warehouses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/PotentialWarehouses_Costs.csv', dtype=str, keep_default_na=False)
stores_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/Stores_Demands.csv', dtype=str, keep_default_na=False)
trans_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/TransportationCost.csv', dtype=str, keep_default_na=False)
warehouse_ids = warehouses_df['Warehouse (i)'].astype(int).tolist()
opening_cost = dict(zip(warehouses_df['Warehouse (i)'].astype(int), warehouses_df['Opening Cost (fi)'].astype(int)))
capacity = dict(zip(warehouses_df['Warehouse (i)'].astype(int), warehouses_df['Capacity (units)'].astype(int)))
store_ids = stores_df['Store (j)'].astype(int).tolist()
demand = dict(zip(stores_df['Store (j)'].astype(int), stores_df['Demand (units, dj)'].astype(int)))

def warehouse_id_to_label(i):
    return f'W{i}'
c_ij = {}
for i in warehouse_ids:
    row_label = warehouse_id_to_label(i)
    row = trans_cost_df[trans_cost_df['Unnamed: 0'].str.strip() == row_label]
    if row.empty:
        raise ValueError(f'TransportationCost.csv missing row for warehouse label {row_label}')
    row = row.iloc[0]
    for j in store_ids:
        col_label = warehouse_id_to_label(j)
        if col_label not in trans_cost_df.columns:
            raise ValueError(f'TransportationCost.csv missing column for store label {col_label}')
        try:
            c_ij[i, j] = int(row[col_label])
        except Exception as e:
            raise ValueError(f'Invalid transportation cost for warehouse {i}, store {j}: {e}')
m = gp.Model('CapacitatedFacilityLocation')
y_vars = m.addVars(warehouse_ids, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(warehouse_ids, store_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((opening_cost[i] * y_vars[i] for i in warehouse_ids)) + gp.quicksum((c_ij[i, j] * x_vars[i, j] for i in warehouse_ids for j in store_ids)), gp.GRB.MINIMIZE)
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
            print(f'  Warehouse {i}: OPEN (Opening cost: {opening_cost[i]}, Capacity: {capacity[i]})')
    print('--- Shipments ---')
    for i in warehouse_ids:
        for j in store_ids:
            if x_vars[i, j].X > 1e-06:
                print(f'  Ship {x_vars[i, j].X:.2f} units from Warehouse {i} to Store {j} (Cost per unit: {c_ij[i, j]})')
else:
    print(f'No optimal solution found. Status: {m.status}')