import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csvs = [('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv', ['customer', 'demand']), ('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv', ['region', 'supply_capacity']), ('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv', ['Unnamed: 0', 'D1', 'D2', 'D3', 'D4', 'D5'])]

    def read_csv_with_fallback(path, **kwargs):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    demand_df = read_csv_with_fallback(csvs[0][0])
    if set(csvs[0][1]) - set(demand_df.columns):
        raise ValueError(f'Missing columns in {csvs[0][0]}')
    demand_df = demand_df[csvs[0][1]]
    supply_df = read_csv_with_fallback(csvs[1][0])
    if set(csvs[1][1]) - set(supply_df.columns):
        raise ValueError(f'Missing columns in {csvs[1][0]}')
    supply_df = supply_df[csvs[1][1]]
    cost_df = read_csv_with_fallback(csvs[2][0])
    if set(csvs[2][1]) - set(cost_df.columns):
        raise ValueError(f'Missing columns in {csvs[2][0]}')
    warehouses = [str(r) for r in supply_df['region']]
    stores = [str(c) for c in demand_df['customer']]
    demand = {}
    for (_, row) in demand_df.iterrows():
        key = str(row['customer'])
        if key in demand:
            demand[key] += float(row['demand'])
        else:
            demand[key] = float(row['demand'])
    supply_capacity = {}
    for (_, row) in supply_df.iterrows():
        key = str(row['region'])
        if key in supply_capacity:
            supply_capacity[key] += float(row['supply_capacity'])
        else:
            supply_capacity[key] = float(row['supply_capacity'])
    cost = {}
    for (_, row) in cost_df.iterrows():
        warehouse = str(row['Unnamed: 0'])
        if warehouse not in warehouses:
            continue
        cost[warehouse] = {}
        for store in stores:
            if store not in cost_df.columns:
                raise ValueError(f'Store {store} not found in cost matrix columns.')
            val = row[store]
            if pd.isnull(val):
                raise ValueError(f'Missing cost for ({warehouse},{store})')
            cost[warehouse][store] = float(val)
    for i in warehouses:
        if i not in cost:
            raise ValueError(f'Warehouse {i} missing in cost matrix.')
        for j in stores:
            if j not in cost[i]:
                raise ValueError(f'Cost for ({i},{j}) missing.')
    for j in stores:
        if j not in demand:
            raise ValueError(f'Demand for store {j} missing.')
    for i in warehouses:
        if i not in supply_capacity:
            raise ValueError(f'Supply for warehouse {i} missing.')
    m = gp.Model('GreenMart_TP')
    m.Params.MIPGap = 0.0001
    keys = [(i, j) for i in warehouses for j in stores]
    x = m.addVars(keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for (i, j) in keys)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in warehouses)) >= demand[j] for j in stores), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in stores)) <= supply_capacity[i] for i in warehouses), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()