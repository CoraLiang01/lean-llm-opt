import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_with_encodings(path):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/cost.csv'
    warehouse_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/warehouse.csv'
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/demand.csv'
    cost_df = read_csv_with_encodings(cost_path)
    warehouse_df = read_csv_with_encodings(warehouse_path)
    demand_df = read_csv_with_encodings(demand_path)
    warehouse_ids = warehouse_df['Warehouse ID'].tolist()
    customer_ids = [col for col in cost_df.columns if col != 'Warehouse ID']
    cost = {}
    for (_, row) in cost_df.iterrows():
        wid = row['Warehouse ID']
        cost[wid] = {}
        for cid in customer_ids:
            val = row[cid]
            if val == '':
                raise ValueError(f'Missing cost for warehouse {wid}, customer {cid}')
            cost[wid][cid] = float(val)
    fixed_cost = {}
    capacity = {}
    for (_, row) in warehouse_df.iterrows():
        wid = row['Warehouse ID']
        if row['Fixed_Cost'] == '' or row['Capacity'] == '':
            raise ValueError(f'Missing fixed cost or capacity for warehouse {wid}')
        fixed_cost[wid] = float(row['Fixed_Cost'])
        capacity[wid] = float(row['Capacity'])
    demand = {}
    for (_, row) in demand_df.iterrows():
        cid = row['Customer ID']
        if row['Demand'] == '':
            raise ValueError(f'Missing demand for customer {cid}')
        demand[cid] = float(row['Demand'])
    for wid in warehouse_ids:
        if wid not in cost:
            raise ValueError(f'Warehouse {wid} missing in cost data')
        if wid not in fixed_cost or wid not in capacity:
            raise ValueError(f'Warehouse {wid} missing in fixed cost or capacity data')
        for cid in customer_ids:
            if cid not in cost[wid]:
                raise ValueError(f'Cost missing for warehouse {wid}, customer {cid}')
    for cid in customer_ids:
        if cid not in demand:
            raise ValueError(f'Demand missing for customer {cid}')
    x_keys = [(wid, cid) for wid in warehouse_ids for cid in customer_ids]
    y_keys = warehouse_ids
    m = gp.Model('UFLP4')
    quantity_vars = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    open_vars = m.addVars(y_keys, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[wid][cid] * quantity_vars[wid, cid] for (wid, cid) in x_keys)) + gp.quicksum((fixed_cost[wid] * open_vars[wid] for wid in y_keys)), GRB.MINIMIZE)
    for cid in customer_ids:
        m.addConstr(gp.quicksum((quantity_vars[wid, cid] for wid in warehouse_ids)) == demand[cid], name=f'demand_{cid}')
    for wid in warehouse_ids:
        m.addConstr(gp.quicksum((quantity_vars[wid, cid] for cid in customer_ids)) <= capacity[wid], name=f'cap_{wid}')
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