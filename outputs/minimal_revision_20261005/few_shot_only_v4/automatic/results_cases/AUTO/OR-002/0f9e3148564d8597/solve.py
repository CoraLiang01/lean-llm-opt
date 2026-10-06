import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csvs = [('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/customer_demand.csv', ['customer', 'demand']), ('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/supply_capacity.csv', ['Unnamed: 0', 'supply_capacity']), ('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/transportation_costs.csv', ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12'])]

    def read_csv_with_fallback(path, **kwargs):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with supported encodings.')
    df_demand = read_csv_with_fallback(csvs[0][0])
    if not set(csvs[0][1]).issubset(df_demand.columns):
        raise ValueError(f'Missing columns in {csvs[0][0]}')
    customers = df_demand['customer'].astype(str).tolist()
    demand = df_demand.set_index('customer')['demand'].to_dict()
    df_supply = read_csv_with_fallback(csvs[1][0])
    if not set(csvs[1][1]).issubset(df_supply.columns):
        raise ValueError(f'Missing columns in {csvs[1][0]}')
    stores = df_supply['Unnamed: 0'].astype(str).tolist()
    supply_capacity = df_supply.set_index('Unnamed: 0')['supply_capacity'].to_dict()
    df_cost = read_csv_with_fallback(csvs[2][0])
    if not set(csvs[2][1]).issubset(df_cost.columns):
        raise ValueError(f'Missing columns in {csvs[2][0]}')
    cost = {}
    for (_, row) in df_cost.iterrows():
        store = str(row['Unnamed: 0'])
        if store not in stores:
            continue
        cost[store] = {}
        for cust in customers:
            if cust not in df_cost.columns:
                raise ValueError(f'Customer {cust} not found in cost columns')
            cost[store][cust] = float(row[cust])
    for i in stores:
        if i not in cost:
            raise ValueError(f'Missing cost row for store {i}')
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Missing cost coefficient for ({i},{j})')
    for j in customers:
        if j not in demand:
            raise ValueError(f'Missing demand for customer {j}')
    for i in stores:
        if i not in supply_capacity:
            raise ValueError(f'Missing supply capacity for store {i}')
    m = gp.Model('Walmart_TP')
    m.Params.MIPGap = 0.0001
    x = m.addVars(stores, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in stores for j in customers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in stores)) >= demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= supply_capacity[i] for i in stores), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()