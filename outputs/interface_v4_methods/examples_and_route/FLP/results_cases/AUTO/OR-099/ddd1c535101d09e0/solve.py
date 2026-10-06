import gurobipy as gp
import pandas as pd
import numpy as np
import re

def int_to_W(n):
    return f'W{int(n)}'

def W_to_int(w):
    m = re.match('W(\\d+)', str(w).strip())
    if not m:
        raise ValueError(f'Invalid warehouse/store label: {w}')
    return int(m.group(1))
warehouses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/PotentialWarehouses_Costs.csv', sep=',')
stores_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/Stores_Demands.csv', sep=',')
trans_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/TransportationCost.csv', sep=',')
warehouse_ids = warehouses_df['Warehouse (i)'].astype(int).tolist()
store_ids = stores_df['Store (j)'].astype(int).tolist()
opening_cost = warehouses_df.set_index('Warehouse (i)')['Opening Cost (fi)'].astype(float).to_dict()
capacity = warehouses_df.set_index('Warehouse (i)')['Capacity (units)'].astype(float).to_dict()
demand = stores_df.set_index('Store (j)')['Demand (units, dj)'].astype(float).to_dict()
trans_cost_df = trans_cost_df.rename(columns=lambda x: x.strip())
trans_cost_df['Unnamed: 0'] = trans_cost_df['Unnamed: 0'].str.strip()
warehouse_labels = [int_to_W(i) for i in warehouse_ids]
store_labels = [int_to_W(j) for j in store_ids]
if not set(warehouse_labels).issubset(set(trans_cost_df.columns)):
    missing = set(warehouse_labels) - set(trans_cost_df.columns)
    raise ValueError(f'Missing warehouse columns in TransportationCost.csv: {missing}')
if not set(store_labels).issubset(set(trans_cost_df['Unnamed: 0'])):
    missing = set(store_labels) - set(trans_cost_df['Unnamed: 0'])
    raise ValueError(f'Missing store rows in TransportationCost.csv: {missing}')
c_ij = {}
for i in warehouse_ids:
    row_label = int_to_W(i)
    row = trans_cost_df[trans_cost_df['Unnamed: 0'] == row_label]
    if row.empty:
        raise ValueError(f'Row for warehouse/store {row_label} not found in TransportationCost.csv')
    for j in store_ids:
        col_label = int_to_W(j)
        cost = float(row.iloc[0][col_label])
        c_ij[i, j] = cost
m = gp.Model('CapacitatedFacilityLocation')
y = m.addVars(warehouse_ids, vtype=gp.GRB.BINARY, name='')
x = m.addVars(warehouse_ids, store_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((opening_cost[i] * y[i] for i in warehouse_ids)) + gp.quicksum((c_ij[i, j] * x[i, j] for i in warehouse_ids for j in store_ids)), gp.GRB.MINIMIZE)
for j in store_ids:
    m.addConstr(gp.quicksum((x[i, j] for i in warehouse_ids)) == demand[j], name=f'demand_{j}')
for i in warehouse_ids:
    m.addConstr(gp.quicksum((x[i, j] for j in store_ids)) <= capacity[i] * y[i], name=f'capacity_{i}')
m.optimize()