import pandas as pd
import gurobipy as gp
from gurobipy import GRB
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/cost.csv', dtype=str, keep_default_na=False)
warehouse_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/warehouse.csv', dtype=str, keep_default_na=False)
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/demand.csv', dtype=str, keep_default_na=False)
warehouses = [row['Warehouse ID'].strip() for _, row in warehouse_df.iterrows()]
customers = [row['Customer ID'].strip() for _, row in demand_df.iterrows()]
demand = {}
for _, row in demand_df.iterrows():
    cid = row['Customer ID'].strip()
    try:
        demand[cid] = int(row['Demand'])
    except Exception:
        raise ValueError(f'Invalid demand for customer {cid}')
fixed_cost = {}
capacity = {}
for _, row in warehouse_df.iterrows():
    wid = row['Warehouse ID'].strip()
    try:
        fixed_cost[wid] = float(row['Fixed_Cost'])
        capacity[wid] = int(row['Capacity'])
    except Exception:
        raise ValueError(f'Invalid fixed cost or capacity for warehouse {wid}')
cost = {}
cost_df = cost_df.rename(columns=lambda x: x.strip())
for _, row in cost_df.iterrows():
    wid = row['Warehouse ID'].strip()
    if wid not in warehouses:
        continue
    cost[wid] = {}
    for cid in customers:
        try:
            cost[wid][cid] = float(row[cid])
        except Exception:
            raise ValueError(f'Missing or invalid cost for warehouse {wid}, customer {cid}')
if set(cost.keys()) != set(warehouses):
    raise ValueError('Mismatch in warehouse IDs between cost.csv and warehouse.csv')
for wid in warehouses:
    if set(cost[wid].keys()) != set(customers):
        raise ValueError(f'Mismatch in customer IDs for warehouse {wid} in cost.csv')
if set(demand.keys()) != set(customers):
    raise ValueError('Mismatch in customer IDs between demand.csv and cost.csv')
if set(fixed_cost.keys()) != set(warehouses) or set(capacity.keys()) != set(warehouses):
    raise ValueError('Mismatch in warehouse IDs between warehouse.csv and cost.csv')
m = gp.Model('UFLP4')
x_vars = m.addVars(warehouses, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(warehouses, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[wid][cid] * x_vars[wid, cid] for wid in warehouses for cid in customers)) + gp.quicksum((fixed_cost[wid] * y_vars[wid] for wid in warehouses)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[wid, cid] for wid in warehouses)) == demand[cid] for cid in customers), name='')
m.addConstrs((gp.quicksum((x_vars[wid, cid] for cid in customers)) <= capacity[wid] * y_vars[wid] for wid in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')