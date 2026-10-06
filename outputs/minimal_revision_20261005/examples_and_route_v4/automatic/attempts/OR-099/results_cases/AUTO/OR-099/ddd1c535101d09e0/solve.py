import gurobipy as gp
import pandas as pd
import numpy as np
import re
warehouses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/PotentialWarehouses_Costs.csv', sep=',')
stores_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/Stores_Demands.csv', sep=',')
trans_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/TransportationCost.csv', sep=',')
warehouse_ids = warehouses_df['Warehouse (i)'].astype(int).tolist()
opening_cost = warehouses_df.set_index('Warehouse (i)')['Opening Cost (fi)'].astype(float).to_dict()
capacity = warehouses_df.set_index('Warehouse (i)')['Capacity (units)'].astype(float).to_dict()
store_ids = stores_df['Store (j)'].astype(int).tolist()
demand = stores_df.set_index('Store (j)')['Demand (units, dj)'].astype(float).to_dict()

def warehouse_id_to_label(i):
    return f'W{i}'

def store_id_to_label(j):
    return f'W{j}'
expected_warehouse_labels = [warehouse_id_to_label(i) for i in warehouse_ids]
expected_store_labels = [store_id_to_label(j) for j in store_ids]
row_labels = trans_cost_df['Unnamed: 0'].astype(str).tolist()
if set(expected_warehouse_labels) != set(row_labels):
    raise ValueError(f'Mismatch between warehouse IDs and transportation cost row labels: {set(expected_warehouse_labels)} vs {set(row_labels)}')
cost_columns = [col for col in trans_cost_df.columns if col != 'Unnamed: 0']
if set(expected_store_labels) != set(cost_columns):
    raise ValueError(f'Mismatch between store IDs and transportation cost column labels: {set(expected_store_labels)} vs {set(cost_columns)}')
c_ij = {}
for (_, row) in trans_cost_df.iterrows():
    i_label = row['Unnamed: 0']
    m = re.fullmatch('W(\\d+)', str(i_label).strip())
    if not m:
        raise ValueError(f'Unexpected warehouse label in TransportationCost.csv: {i_label}')
    i = int(m.group(1))
    for j in store_ids:
        j_label = store_id_to_label(j)
        if j_label not in row:
            raise ValueError(f'Store label {j_label} not found in TransportationCost.csv columns')
        c_ij[i, j] = float(row[j_label])
for i in warehouse_ids:
    if i not in opening_cost or i not in capacity:
        raise ValueError(f'Missing opening cost or capacity for warehouse {i}')
for j in store_ids:
    if j not in demand:
        raise ValueError(f'Missing demand for store {j}')
for i in warehouse_ids:
    for j in store_ids:
        if (i, j) not in c_ij:
            raise ValueError(f'Missing transportation cost for warehouse {i} to store {j}')
m = gp.Model('CapacitatedFacilityLocation')
y = m.addVars(warehouse_ids, vtype=gp.GRB.BINARY, name='')
x = m.addVars([(i, j) for i in warehouse_ids for j in store_ids], vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((opening_cost[i] * y[i] for i in warehouse_ids)) + gp.quicksum((c_ij[i, j] * x[i, j] for i in warehouse_ids for j in store_ids)), gp.GRB.MINIMIZE)
for j in store_ids:
    m.addConstr(gp.quicksum((x[i, j] for i in warehouse_ids)) == demand[j], name=f'demand_{j}')
for i in warehouse_ids:
    m.addConstr(gp.quicksum((x[i, j] for j in store_ids)) <= capacity[i] * y[i], name=f'capacity_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in warehouse_ids:
        print(f'y[{i}] {y[i].VarName} {y[i].X}')
    for i in warehouse_ids:
        for j in store_ids:
            print(f'x[{i},{j}] {x[i, j].VarName} {x[i, j].X}')
else:
    print(f'Solver status: {m.status}')