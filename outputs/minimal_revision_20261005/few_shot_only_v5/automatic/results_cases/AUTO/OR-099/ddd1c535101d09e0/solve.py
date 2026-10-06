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
            warehouses_df = pd.read_csv(warehouse_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {warehouse_path} with tried encodings.')
    for enc in encodings:
        try:
            stores_df = pd.read_csv(store_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {store_path} with tried encodings.')
    for enc in encodings:
        try:
            cost_df = pd.read_csv(cost_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {cost_path} with tried encodings.')
    warehouses = warehouses_df['Warehouse (i)'].astype(str).tolist()
    f = dict(zip(warehouses_df['Warehouse (i)'].astype(str), warehouses_df['Opening Cost (fi)']))
    K = dict(zip(warehouses_df['Warehouse (i)'].astype(str), warehouses_df['Capacity (units)']))
    stores = stores_df['Store (j)'].astype(str).tolist()
    d = dict(zip(stores_df['Store (j)'].astype(str), stores_df['Demand (units, dj)']))
    if 'Unnamed: 0' in cost_df.columns:
        cost_df = cost_df.rename(columns={'Unnamed: 0': 'Store (j)'})
    cost_df['Store (j)'] = cost_df['Store (j)'].astype(str)
    cost_df = cost_df.set_index('Store (j)')
    missing_warehouses = set(warehouses) - set(cost_df.columns)
    missing_stores = set(stores) - set(cost_df.index)
    if missing_warehouses:
        raise ValueError(f'Missing warehouses in TransportationCost.csv: {missing_warehouses}')
    if missing_stores:
        raise ValueError(f'Missing stores in TransportationCost.csv: {missing_stores}')
    c = {}
    for i in warehouses:
        c[i] = {}
        for j in stores:
            try:
                c[i][j] = float(cost_df.loc[j, i])
            except Exception:
                raise ValueError(f'Missing transportation cost for warehouse {i}, store {j}')
    for i in warehouses:
        if i not in f or i not in K:
            raise ValueError(f'Missing opening cost or capacity for warehouse {i}')
    for j in stores:
        if j not in d:
            raise ValueError(f'Missing demand for store {j}')
    m = gp.Model('UFLP14')
    x_keys = [(i, j) for i in warehouses for j in stores]
    x = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(warehouses, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((f[i] * y[i] for i in warehouses)) + gp.quicksum((c[i][j] * x[i, j] for i in warehouses for j in stores)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in warehouses)) == d[j] for j in stores), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in stores)) <= K[i] * y[i] for i in warehouses), name='')
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