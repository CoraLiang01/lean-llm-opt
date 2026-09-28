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

def id_to_wstr(idx):
    return f'W{int(idx)}'
expected_cols = [id_to_wstr(i) for i in warehouse_ids]
if not all((col in transport_df.columns for col in expected_cols)):
    raise ValueError('TransportationCost.csv missing expected warehouse columns.')
if not all((isinstance(x, str) and re.fullmatch('W\\d+', x.strip()) for x in transport_df['Unnamed: 0'])):
    raise ValueError("TransportationCost.csv 'Unnamed: 0' must be warehouse labels like 'W1', 'W2', ...")
c_ij = {}
for i in warehouse_ids:
    row_label = id_to_wstr(i)
    row = transport_df[transport_df['Unnamed: 0'].str.strip() == row_label]
    if row.empty:
        raise ValueError(f'TransportationCost.csv missing row for warehouse {row_label}')
    row = row.iloc[0]
    for j in store_ids:
        col_label = id_to_wstr(j)
        if col_label not in row:
            raise ValueError(f'TransportationCost.csv missing column {col_label}')
        c_ij[i, j] = float(row[col_label])
m = gp.Model('CapacitatedFacilityLocation')
y = m.addVars(warehouse_ids, vtype=gp.GRB.BINARY, name='')
x = m.addVars(warehouse_ids, store_ids, vtype=gp.GRB.CONTINUOUS, name='')
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
            print(f'  Warehouse {i}: OPEN (capacity {int(capacity[i])}, opening cost {int(opening_cost[i])})')
    print('--- Store Assignments ---')
    for j in store_ids:
        print(f'Store {j} (demand {int(demand[j])}):')
        for i in warehouse_ids:
            if x[i, j].X > 1e-06:
                print(f'  Supplied {x[i, j].X:.2f} units from Warehouse {i} (cost per unit {c_ij[i, j]})')
else:
    print(f'No optimal solution found. Status: {m.status}')