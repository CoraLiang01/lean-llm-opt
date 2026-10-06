import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/customer_demand.csv'
    supply_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/supply_capacity.csv'
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/transportation_costs.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            customer_demand = pd.read_csv(demand_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Failed to read {demand_path} with tried encodings.')
    for enc in encodings:
        try:
            supply_capacity = pd.read_csv(supply_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Failed to read {supply_path} with tried encodings.')
    for enc in encodings:
        try:
            transportation_costs = pd.read_csv(cost_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Failed to read {cost_path} with tried encodings.')
    customers = customer_demand['customer'].astype(str).tolist()
    stores = supply_capacity['Unnamed: 0'].astype(str).tolist()
    demand = {}
    for (_, row) in customer_demand.iterrows():
        cust = str(row['customer'])
        if cust in demand:
            demand[cust] += float(row['demand'])
        else:
            demand[cust] = float(row['demand'])
    supply = {}
    for (_, row) in supply_capacity.iterrows():
        store = str(row['Unnamed: 0'])
        if store in supply:
            supply[store] += float(row['supply_capacity'])
        else:
            supply[store] = float(row['supply_capacity'])
    cost = {}
    cost_cols = [col for col in transportation_costs.columns if col != 'Unnamed: 0']
    for (_, row) in transportation_costs.iterrows():
        store = str(row['Unnamed: 0'])
        cost[store] = {}
        for cust in cost_cols:
            cost[store][cust] = float(row[cust])
    missing_stores = set(stores) - set(cost.keys())
    missing_customers = set(customers) - set(cost_cols)
    if missing_stores:
        raise ValueError(f'Stores missing in cost matrix: {missing_stores}')
    if missing_customers:
        raise ValueError(f'Customers missing in cost matrix: {missing_customers}')
    keys = [(i, j) for i in stores for j in customers]
    m = gp.Model('Walmart_Transportation')
    x = m.addVars(keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for (i, j) in keys)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in stores)) >= demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= supply[i] for i in stores), name='')
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