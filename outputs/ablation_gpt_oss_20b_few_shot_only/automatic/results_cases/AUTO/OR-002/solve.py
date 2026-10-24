import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    path_demand = 'Test_Dataset/Large-scale-or/TP_testing/TP2/customer_demand.csv'
    path_supply = 'Test_Dataset/Large-scale-or/TP_testing/TP2/supply_capacity.csv'
    path_cost = 'Test_Dataset/Large-scale-or/TP_testing/TP2/transportation_costs.csv'
    df_demand = pd.read_csv(path_demand, dtype=str, keep_default_na=False, header=None)
    if df_demand.shape[1] == 1:
        df_demand = df_demand.iloc[1:].reset_index(drop=True)
    if df_demand.shape[1] < 2:
        raise ValueError('customer_demand.csv must have at least two columns')
    df_demand.columns = ['customer', 'demand']
    df_demand['customer'] = df_demand['customer'].str.strip()
    df_demand['demand'] = pd.to_numeric(df_demand['demand'].str.strip(), errors='raise')
    demand = dict(zip(df_demand['customer'], df_demand['demand']))
    df_supply = pd.read_csv(path_supply, dtype=str, keep_default_na=False, header=0)
    df_supply['Unnamed: 0'] = df_supply['Unnamed: 0'].str.strip()
    df_supply['supply_capacity'] = pd.to_numeric(df_supply['supply_capacity'].str.strip(), errors='raise')
    supply = dict(zip(df_supply['Unnamed: 0'], df_supply['supply_capacity']))
    df_cost = pd.read_csv(path_cost, dtype=str, keep_default_na=False, header=0)
    df_cost['Unnamed: 0'] = df_cost['Unnamed: 0'].str.strip()
    cost_cols = [col for col in df_cost.columns if col != 'Unnamed: 0']
    for col in cost_cols:
        df_cost[col] = pd.to_numeric(df_cost[col].str.strip(), errors='raise')
    cost = {}
    for (_, row) in df_cost.iterrows():
        store = row['Unnamed: 0']
        cost[store] = {col: row[col] for col in cost_cols}
    stores = set(supply.keys())
    customers = set(demand.keys())
    if not stores.issubset(set(cost.keys())):
        missing = stores - set(cost.keys())
        raise ValueError(f'Missing cost data for stores: {missing}')
    if not customers.issubset(set(cost_cols)):
        missing = customers - set(cost_cols)
        raise ValueError(f'Missing cost data for customers: {missing}')
    m = gurobipy.Model('TransportationProblem')
    m.setParam('MIPGap', 1e-06)
    keys = [(i, j) for i in stores for j in customers]
    x = m.addVars(keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((cost[i][j] * x[i, j] for i in stores for j in customers)), GRB.MINIMIZE)
    for j in customers:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in stores)) >= demand[j], name=f'demand_{j}')
    for i in stores:
        m.addConstr(gurobipy.quicksum((x[i, j] for j in customers)) <= supply[i], name=f'supply_{i}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()