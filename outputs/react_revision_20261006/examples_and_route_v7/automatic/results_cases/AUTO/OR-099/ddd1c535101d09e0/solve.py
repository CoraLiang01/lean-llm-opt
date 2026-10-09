import gurobipy as gp
import pandas as pd
import numpy as np
import re
warehouses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/PotentialWarehouses_Costs.csv', dtype=str, keep_default_na=False)
stores_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/Stores_Demands.csv', dtype=str, keep_default_na=False)
trans_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/TransportationCost.csv', dtype=str, keep_default_na=False)
warehouse_ids = warehouses_df['Warehouse (i)'].astype(int).tolist()
opening_cost = warehouses_df.set_index(warehouses_df['Warehouse (i)'].astype(int))['Opening Cost (fi)'].astype(int).to_dict()
capacity = warehouses_df.set_index(warehouses_df['Warehouse (i)'].astype(int))['Capacity (units)'].astype(int).to_dict()
store_ids = stores_df['Store (j)'].astype(int).tolist()
demand = stores_df.set_index(stores_df['Store (j)'].astype(int))['Demand (units, dj)'].astype(int).to_dict()
warehouse_id_to_label = {wid: f'W{wid}' for wid in warehouse_ids}
store_id_to_label = {sid: f'W{sid}' for sid in store_ids}
cost_row_labels = trans_cost_df['Unnamed: 0'].tolist()
cost_col_labels = [col for col in trans_cost_df.columns if col != 'Unnamed: 0']
missing_rows = set(warehouse_id_to_label.values()) - set(cost_row_labels)
missing_cols = set(store_id_to_label.values()) - set(cost_col_labels)
if missing_rows:
    raise ValueError(f'Missing warehouse rows in TransportationCost.csv: {missing_rows}')
if missing_cols:
    raise ValueError(f'Missing store columns in TransportationCost.csv: {missing_cols}')
c_ij = {}
for i in warehouse_ids:
    row_label = warehouse_id_to_label[i]
    row = trans_cost_df[trans_cost_df['Unnamed: 0'] == row_label]
    if row.empty:
        raise ValueError(f'Warehouse row {row_label} not found in TransportationCost.csv')
    for j in store_ids:
        col_label = store_id_to_label[j]
        val = row.iloc[0][col_label]
        try:
            c_ij[i, j] = int(val)
        except Exception:
            raise ValueError(f'Invalid transportation cost for warehouse {i}, store {j}: {val}')
for i in warehouse_ids:
    if i not in opening_cost or i not in capacity:
        raise ValueError(f'Missing opening cost or capacity for warehouse {i}')
for j in store_ids:
    if j not in demand:
        raise ValueError(f'Missing demand for store {j}')
for i in warehouse_ids:
    for j in store_ids:
        if (i, j) not in c_ij:
            raise ValueError(f'Missing transportation cost for warehouse {i}, store {j}')

def solve_uflp(warehouse_ids, store_ids, opening_cost, capacity, demand, c_ij):
    m = gp.Model('UFLP')
    m.Params.MIPGap = 0.0001
    y_vars = m.addVars(warehouse_ids, vtype=gp.GRB.BINARY, name='')
    x_keys = [(i, j) for i in warehouse_ids for j in store_ids]
    x_vars = m.addVars(x_keys, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(gp.quicksum((opening_cost[i] * y_vars[i] for i in warehouse_ids)) + gp.quicksum((c_ij[i, j] * x_vars[i, j] for i in warehouse_ids for j in store_ids)), gp.GRB.MINIMIZE)
    for j in store_ids:
        m.addConstr(gp.quicksum((x_vars[i, j] for i in warehouse_ids)) == demand[j], name=f'demand_{j}')
    for i in warehouse_ids:
        m.addConstr(gp.quicksum((x_vars[i, j] for j in store_ids)) <= capacity[i] * y_vars[i], name=f'capacity_{i}')
    m.optimize()
    return m
m = solve_uflp(warehouse_ids, store_ids, opening_cost, capacity, demand, c_ij)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')