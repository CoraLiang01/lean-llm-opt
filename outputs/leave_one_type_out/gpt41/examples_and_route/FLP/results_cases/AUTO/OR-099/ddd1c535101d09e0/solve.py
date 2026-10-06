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
opening_cost = warehouses_df.set_index('Warehouse (i)')['Opening Cost (fi)'].astype(int).to_dict()
capacity = warehouses_df.set_index('Warehouse (i)')['Capacity (units)'].astype(int).to_dict()
store_ids = stores_df['Store (j)'].astype(int).tolist()
demand = stores_df.set_index('Store (j)')['Demand (units, dj)'].astype(int).to_dict()

def wlabel_to_id(wlabel):
    m = re.match('W(\\d+)', str(wlabel).strip())
    if not m:
        raise ValueError(f'Invalid warehouse/store label: {wlabel}')
    return int(m.group(1))
col_labels = [col for col in transport_df.columns if col != 'Unnamed: 0']
col_ids = [wlabel_to_id(col) for col in col_labels]
row_labels = transport_df['Unnamed: 0'].tolist()
row_ids = [wlabel_to_id(row) for row in row_labels]
if sorted(warehouse_ids) != sorted(row_ids):
    raise ValueError('Mismatch between warehouse IDs in PotentialWarehouses_Costs.csv and TransportationCost.csv rows')
if sorted(store_ids) != sorted(col_ids):
    raise ValueError('Mismatch between store IDs in Stores_Demands.csv and TransportationCost.csv columns')
c_ij = {}
for row_idx, i in enumerate(row_ids):
    for col_idx, j in enumerate(col_ids):
        c_ij[i, j] = int(transport_df.iloc[row_idx, col_idx + 1])
m = gp.Model('CapacitatedFacilityLocation')
y = m.addVars(warehouse_ids, vtype=gp.GRB.BINARY, name='')
x = m.addVars(warehouse_ids, store_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
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
            print(f'Warehouse {i}: OPEN (capacity {capacity[i]}, opening cost {opening_cost[i]})')
    print('--- Store Assignments ---')
    for j in store_ids:
        print(f'Store {j} (demand {demand[j]}):')
        for i in warehouse_ids:
            if x[i, j].X > 1e-06:
                print(f'  Supplied {int(round(x[i, j].X))} units from Warehouse {i} (cost per unit {c_ij[i, j]})')
else:
    print(f'No optimal solution found. Status: {m.status}')