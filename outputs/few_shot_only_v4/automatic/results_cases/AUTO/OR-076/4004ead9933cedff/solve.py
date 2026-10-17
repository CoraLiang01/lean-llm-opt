import gurobipy as gp
from gurobipy import GRB
import pandas as pd
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/cost.csv', dtype=str, keep_default_na=False)
warehouse_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/warehouse.csv', dtype=str, keep_default_na=False)
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/demand.csv', dtype=str, keep_default_na=False)
warehouses = list(warehouse_df['Warehouse'])
customers = list(demand_df['Customer'])
cost_row_ids = list(cost_df.iloc[:, 0])
cost_col_ids = list(cost_df.columns[1:])
if set(warehouses) != set(cost_row_ids):
    raise ValueError('Mismatch between warehouse identifiers in warehouse.csv and cost.csv')
if set(customers) != set(cost_col_ids):
    raise ValueError('Mismatch between customer identifiers in demand.csv and cost.csv')
cost = {}
for i, wid in enumerate(cost_row_ids):
    row = cost_df.iloc[i, 1:]
    cost[wid] = {}
    for j, cid in enumerate(cost_col_ids):
        try:
            cost[wid][cid] = float(row.iloc[j])
        except Exception:
            raise ValueError(f'Invalid cost entry for warehouse {wid}, customer {cid}')
fixed_cost = {}
capacity = {}
for idx, row in warehouse_df.iterrows():
    wid = row['Warehouse']
    try:
        fixed_cost[wid] = float(row['Fixed Cost'])
        capacity[wid] = float(row['Capacity'])
    except Exception:
        raise ValueError(f'Invalid fixed cost or capacity for warehouse {wid}')
demand = {}
for idx, row in demand_df.iterrows():
    cid = row['Customer']
    try:
        demand[cid] = float(row['Demand'])
    except Exception:
        raise ValueError(f'Invalid demand entry for customer {cid}')
for wid in warehouses:
    if wid not in fixed_cost or wid not in capacity or wid not in cost:
        raise ValueError(f'Missing data for warehouse {wid}')
    for cid in customers:
        if cid not in cost[wid]:
            raise ValueError(f'Missing cost for warehouse {wid}, customer {cid}')
for cid in customers:
    if cid not in demand:
        raise ValueError(f'Missing demand for customer {cid}')
m = gp.Model('UFLP4')
x_vars = m.addVars(warehouses, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(warehouses, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost[wid] * y_vars[wid] for wid in warehouses)) + gp.quicksum((cost[wid][cid] * x_vars[wid, cid] for wid in warehouses for cid in customers)), GRB.MINIMIZE)
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