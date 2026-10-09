import gurobipy as gp
import pandas as pd
import numpy as np
import re
warehouses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/PotentialWarehouses_Costs.csv', dtype=str, keep_default_na=False)
stores_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/Stores_Demands.csv', dtype=str, keep_default_na=False)
transport_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/TransportationCost.csv', dtype=str, keep_default_na=False)
warehouse_ids = warehouses_df['Warehouse (i)'].astype(int).tolist()
opening_cost = dict(zip(warehouses_df['Warehouse (i)'].astype(int), warehouses_df['Opening Cost (fi)'].astype(int)))
capacity = dict(zip(warehouses_df['Warehouse (i)'].astype(int), warehouses_df['Capacity (units)'].astype(int)))
store_ids = stores_df['Store (j)'].astype(int).tolist()
demand = dict(zip(stores_df['Store (j)'].astype(int), stores_df['Demand (units, dj)'].astype(int)))

def id_to_label(idx):
    return f'W{int(idx)}'
warehouse_labels = [id_to_label(i) for i in warehouse_ids]
store_labels = [id_to_label(j) for j in store_ids]
if not all((lab in transport_df.columns for lab in store_labels)):
    missing = [lab for lab in store_labels if lab not in transport_df.columns]
    raise ValueError(f'Missing store columns in TransportationCost.csv: {missing}')
if not all((lab in transport_df['Unnamed: 0'].values for lab in warehouse_labels)):
    missing = [lab for lab in warehouse_labels if lab not in transport_df['Unnamed: 0'].values]
    raise ValueError(f'Missing warehouse rows in TransportationCost.csv: {missing}')
transport_df_indexed = transport_df.set_index('Unnamed: 0')
c_ij = {}
for i in warehouse_ids:
    row_label = id_to_label(i)
    for j in store_ids:
        col_label = id_to_label(j)
        val = transport_df_indexed.at[row_label, col_label]
        try:
            c_ij[i, j] = int(val)
        except Exception:
            raise ValueError(f'Invalid transportation cost for warehouse {i}, store {j}: {val}')
m = gp.Model('CapacitatedFacilityLocation')
y_vars = m.addVars(warehouse_ids, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(warehouse_ids, store_ids, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((opening_cost[i] * y_vars[i] for i in warehouse_ids)) + gp.quicksum((c_ij[i, j] * x_vars[i, j] for i in warehouse_ids for j in store_ids)), gp.GRB.MINIMIZE)
for j in store_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in warehouse_ids)) == demand[j], name=f'demand_{j}')
for i in warehouse_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in store_ids)) <= capacity[i] * y_vars[i], name=f'capacity_{i}')
for i in warehouse_ids:
    for j in store_ids:
        m.addConstr(x_vars[i, j] <= capacity[i] * y_vars[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Warehouses Opened ---')
    for i in warehouse_ids:
        if y_vars[i].X > 0.5:
            print(f'  Warehouse {i}: OPEN (capacity {capacity[i]}, opening cost {opening_cost[i]})')
    print('--- Assignment of Shipments ---')
    for i in warehouse_ids:
        for j in store_ids:
            if x_vars[i, j].X > 1e-06:
                print(f'  Ship {x_vars[i, j].X:.2f} units from Warehouse {i} to Store {j} (cost per unit: {c_ij[i, j]})')
else:
    print(f'No optimal solution found. Status: {m.status}')