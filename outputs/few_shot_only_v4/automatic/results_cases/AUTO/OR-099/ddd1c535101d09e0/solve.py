import pandas as pd
import gurobipy as gp
from gurobipy import GRB
warehouse_csv = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/PotentialWarehouses_Costs.csv'
demand_csv = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/Stores_Demands.csv'
transport_csv = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/TransportationCost.csv'
warehouses_df = pd.read_csv(warehouse_csv, dtype=str, keep_default_na=False)
if not {'Warehouse', 'OpeningCost', 'Capacity'}.issubset(warehouses_df.columns):
    raise ValueError('PotentialWarehouses_Costs.csv must have columns: Warehouse, OpeningCost, Capacity')
warehouses = warehouses_df['Warehouse'].tolist()
f_i = {}
u_i = {}
for idx, row in warehouses_df.iterrows():
    wid = row['Warehouse']
    try:
        f_i[wid] = float(row['OpeningCost'])
        u_i[wid] = float(row['Capacity'])
    except Exception as e:
        raise ValueError(f'Invalid OpeningCost or Capacity for warehouse {wid}: {e}')
demand_df = pd.read_csv(demand_csv, dtype=str, keep_default_na=False)
if not {'Store', 'Demand'}.issubset(demand_df.columns):
    raise ValueError('Stores_Demands.csv must have columns: Store, Demand')
stores = demand_df['Store'].tolist()
d_j = {}
for idx, row in demand_df.iterrows():
    sid = row['Store']
    try:
        d_j[sid] = float(row['Demand'])
    except Exception as e:
        raise ValueError(f'Invalid Demand for store {sid}: {e}')
transport_df = pd.read_csv(transport_csv, dtype=str, keep_default_na=False)
if transport_df.shape[0] != len(warehouses):
    raise ValueError('TransportationCost.csv row count does not match number of warehouses')
if transport_df.shape[1] != len(stores) + 1:
    raise ValueError('TransportationCost.csv column count does not match number of stores + 1 (for warehouse id)')
if transport_df.columns[0] != 'Warehouse':
    raise ValueError("First column of TransportationCost.csv must be 'Warehouse'")
c_ij = {}
for idx, row in transport_df.iterrows():
    wid = row['Warehouse']
    if wid not in warehouses:
        raise ValueError(f'Warehouse {wid} in TransportationCost.csv not found in PotentialWarehouses_Costs.csv')
    c_ij[wid] = {}
    for sid in stores:
        try:
            c_ij[wid][sid] = float(row[sid])
        except Exception as e:
            raise ValueError(f'Invalid transportation cost for warehouse {wid}, store {sid}: {e}')
if set(f_i.keys()) != set(warehouses):
    raise ValueError('Mismatch in warehouse identifiers between cost/capacity and warehouse list')
if set(u_i.keys()) != set(warehouses):
    raise ValueError('Mismatch in warehouse identifiers between capacity and warehouse list')
if set(d_j.keys()) != set(stores):
    raise ValueError('Mismatch in store identifiers between demand and store list')
for wid in warehouses:
    if set(c_ij[wid].keys()) != set(stores):
        raise ValueError(f'Transportation cost data for warehouse {wid} does not cover all stores')
m = gp.Model('UFLP_Warehouse_Selection')
x_vars = m.addVars(warehouses, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(warehouses, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((f_i[wid] * y_vars[wid] for wid in warehouses)) + gp.quicksum((c_ij[wid][sid] * x_vars[wid, sid] for wid in warehouses for sid in stores)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[wid, sid] for wid in warehouses)) == d_j[sid] for sid in stores), name='')
m.addConstrs((gp.quicksum((x_vars[wid, sid] for sid in stores)) <= u_i[wid] * y_vars[wid] for wid in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')