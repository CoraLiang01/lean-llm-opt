import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_robust(path):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with supported encodings.')
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/cost.csv'
    warehouse_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/warehouse.csv'
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/demand.csv'
    cost_df = read_csv_robust(cost_path)
    warehouse_df = read_csv_robust(warehouse_path)
    demand_df = read_csv_robust(demand_path)
    warehouses = cost_df['Warehouse ID'].astype(str).unique().tolist()
    customers = [col for col in cost_df.columns if col != 'Warehouse ID']
    warehouse_ids = warehouse_df['Warehouse ID'].astype(str).unique().tolist()
    if set(warehouses) - set(warehouse_ids):
        raise ValueError('Some warehouses in cost.csv are missing from warehouse.csv')
    customer_ids = demand_df['Customer ID'].astype(str).unique().tolist()
    if set(customers) - set(customer_ids):
        raise ValueError('Some customers in cost.csv are missing from demand.csv')
    cost = {}
    for (_, row) in cost_df.iterrows():
        i = str(row['Warehouse ID'])
        cost[i] = {}
        for j in customers:
            cost[i][j] = float(row[j])
    fixed_cost = {}
    capacity = {}
    for (_, row) in warehouse_df.iterrows():
        i = str(row['Warehouse ID'])
        fixed_cost[i] = float(row['Fixed_Cost'])
        capacity[i] = float(row['Capacity'])
    demand = {}
    for (_, row) in demand_df.iterrows():
        j = str(row['Customer ID'])
        demand[j] = float(row['Demand'])
    for i in warehouses:
        if i not in fixed_cost or i not in capacity:
            raise ValueError(f'Missing warehouse data for {i}')
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Missing cost for warehouse {i}, customer {j}')
    for j in customers:
        if j not in demand:
            raise ValueError(f'Missing demand for customer {j}')
    m = gp.Model('UFLP4')
    m.Params.MIPGap = 0.0001
    x_keys = [(i, j) for i in warehouses for j in customers]
    x = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(warehouses, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in warehouses)) + gp.quicksum((cost[i][j] * x[i, j] for i in warehouses for j in customers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in warehouses)) == demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= capacity[i] * y[i] for i in warehouses), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')