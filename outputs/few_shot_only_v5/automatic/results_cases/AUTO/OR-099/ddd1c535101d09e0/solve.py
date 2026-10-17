import pandas as pd
import gurobipy as gp
from gurobipy import GRB
warehouses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/PotentialWarehouses_Costs.csv', dtype=str, keep_default_na=False)
warehouses_df = warehouses_df.rename(columns=lambda x: x.strip())
warehouse_ids = warehouses_df['Warehouse (i)'].astype(str).tolist()
f = {wid: float(warehouses_df.loc[warehouses_df['Warehouse (i)'] == wid, 'Opening Cost (fi)'].values[0]) for wid in warehouse_ids}
u = {wid: float(warehouses_df.loc[warehouses_df['Warehouse (i)'] == wid, 'Capacity (units)'].values[0]) for wid in warehouse_ids}
stores_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/Stores_Demands.csv', dtype=str, keep_default_na=False)
stores_df = stores_df.rename(columns=lambda x: x.strip())
store_ids = stores_df['Store (j)'].astype(str).tolist()
d = {sid: float(stores_df.loc[stores_df['Store (j)'] == sid, 'Demand (units, dj)'].values[0]) for sid in store_ids}
tcost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/TransportationCost.csv', dtype=str, keep_default_na=False)
tcost_df = tcost_df.rename(columns=lambda x: x.strip())
warehouse_labels = tcost_df['Unnamed: 0'].tolist()
store_labels = [col for col in tcost_df.columns if col != 'Unnamed: 0']
if len(warehouse_ids) != len(warehouse_labels):
    raise ValueError('Mismatch between warehouse IDs and transportation cost warehouse labels')
warehouse_map = {warehouse_labels[i]: warehouse_ids[i] for i in range(len(warehouse_ids))}
store_map = {store_labels[i]: store_ids[i] for i in range(len(store_ids))}
c = {}
for w_label in warehouse_labels:
    w_id = warehouse_map[w_label]
    c[w_id] = {}
    for s_label in store_labels:
        s_id = store_map[s_label]
        val = tcost_df.loc[tcost_df['Unnamed: 0'] == w_label, s_label].values[0]
        c[w_id][s_id] = float(val)
if set(f.keys()) != set(c.keys()) or set(u.keys()) != set(c.keys()):
    raise ValueError('Mismatch in warehouse keys between cost/capacity and transportation cost data')
for w in c:
    if set(c[w].keys()) != set(d.keys()):
        raise ValueError(f'Mismatch in store keys for warehouse {w} between transportation cost and demand data')
m = gp.Model('UFLP_Warehouse_Selection')
x_vars = m.addVars(warehouse_ids, store_ids, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(warehouse_ids, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((f[w] * y_vars[w] for w in warehouse_ids)) + gp.quicksum((c[w][s] * x_vars[w, s] for w in warehouse_ids for s in store_ids)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[w, s] for w in warehouse_ids)) == d[s] for s in store_ids), name='')
m.addConstrs((gp.quicksum((x_vars[w, s] for s in store_ids)) <= u[w] * y_vars[w] for w in warehouse_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')