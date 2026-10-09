import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    warehouse_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/PotentialWarehouses_Costs.csv'
    store_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/Stores_Demands.csv'
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/TransportationCost.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            warehouses_df = pd.read_csv(warehouse_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {warehouse_path} with tried encodings.')
    for enc in encodings:
        try:
            stores_df = pd.read_csv(store_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {store_path} with tried encodings.')
    for enc in encodings:
        try:
            cost_df = pd.read_csv(cost_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {cost_path} with tried encodings.')
    warehouse_ids = warehouses_df['Warehouse (i)'].tolist()
    opening_cost = {}
    capacity = {}
    for (_, row) in warehouses_df.iterrows():
        wid = row['Warehouse (i)']
        try:
            opening_cost[wid] = float(row['Opening Cost (fi)'])
        except Exception:
            raise ValueError(f"Invalid opening cost for warehouse {wid}: {row['Opening Cost (fi)']}")
        try:
            capacity[wid] = float(row['Capacity (units)'])
        except Exception:
            raise ValueError(f"Invalid capacity for warehouse {wid}: {row['Capacity (units)']}")
    store_ids = stores_df['Store (j)'].tolist()
    demand = {}
    for (_, row) in stores_df.iterrows():
        sid = row['Store (j)']
        try:
            demand[sid] = float(row['Demand (units, dj)'])
        except Exception:
            raise ValueError(f"Invalid demand for store {sid}: {row['Demand (units, dj)']}")
    if cost_df.columns[0].casefold().startswith('unnamed'):
        store_col = cost_df.columns[0]
    else:
        store_col = cost_df.columns[0]
    cost = {}
    for (_, row) in cost_df.iterrows():
        sid = row[store_col]
        if sid not in store_ids:
            continue
        cost[sid] = {}
        for wid in warehouse_ids:
            if wid not in row:
                raise ValueError(f'Warehouse {wid} not found in transportation cost columns.')
            try:
                cost[sid][wid] = float(row[wid])
            except Exception:
                raise ValueError(f'Invalid transportation cost for warehouse {wid}, store {sid}: {row[wid]}')
    for sid in store_ids:
        if sid not in cost:
            raise ValueError(f'Missing transportation cost row for store {sid}')
        for wid in warehouse_ids:
            if wid not in cost[sid]:
                raise ValueError(f'Missing transportation cost for warehouse {wid}, store {sid}')
    m = gp.Model('UFLP14')
    quantity_keys = [(wid, sid) for wid in warehouse_ids for sid in store_ids]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    open_vars = m.addVars(warehouse_ids, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((opening_cost[wid] * open_vars[wid] for wid in warehouse_ids)) + gp.quicksum((cost[sid][wid] * quantity_vars[wid, sid] for wid in warehouse_ids for sid in store_ids)), GRB.MINIMIZE)
    for sid in store_ids:
        m.addConstr(gp.quicksum((quantity_vars[wid, sid] for wid in warehouse_ids)) == demand[sid], name=f'demand_{sid}')
    for wid in warehouse_ids:
        m.addConstr(gp.quicksum((quantity_vars[wid, sid] for sid in store_ids)) <= capacity[wid], name=f'capacity_{wid}')
    for wid in warehouse_ids:
        for sid in store_ids:
            m.addConstr(quantity_vars[wid, sid] <= demand[sid] * open_vars[wid], name=f'activation_{wid}_{sid}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()