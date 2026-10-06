import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_with_encodings(path, **kwargs):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/cost.csv'
    warehouse_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/warehouse.csv'
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/demand.csv'
    warehouse_df = read_csv_with_encodings(warehouse_path)
    demand_df = read_csv_with_encodings(demand_path)
    cost_df = read_csv_with_encodings(cost_path)
    warehouses = warehouse_df['Warehouse ID'].astype(str).tolist()
    customers = [col for col in cost_df.columns if col != 'Warehouse ID']
    fixed_cost = {}
    capacity = {}
    for (_, row) in warehouse_df.iterrows():
        wid = str(row['Warehouse ID'])
        fixed_cost[wid] = float(row['Fixed_Cost'])
        capacity[wid] = float(row['Capacity'])
    demand = {}
    for (_, row) in demand_df.iterrows():
        cid = str(row['Customer ID'])
        demand[cid] = float(row['Demand'])
    cost = {}
    for (_, row) in cost_df.iterrows():
        wid = str(row['Warehouse ID'])
        cost[wid] = {}
        for cid in customers:
            cost[wid][cid] = float(row[cid])
    for wid in warehouses:
        if wid not in fixed_cost or wid not in capacity or wid not in cost:
            raise ValueError(f'Missing warehouse data for {wid}')
        for cid in customers:
            if cid not in cost[wid]:
                raise ValueError(f'Missing cost for warehouse {wid}, customer {cid}')
    for cid in customers:
        if cid not in demand:
            raise ValueError(f'Missing demand for customer {cid}')
    m = gp.Model('UFLP4')
    m.Params.MIPGap = 0.0001
    x_keys = [(i, j) for i in warehouses for j in customers]
    x = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(warehouses, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in warehouses)) + gp.quicksum((cost[i][j] * x[i, j] for (i, j) in x_keys)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in warehouses)) == demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= capacity[i] * y[i] for i in warehouses), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()