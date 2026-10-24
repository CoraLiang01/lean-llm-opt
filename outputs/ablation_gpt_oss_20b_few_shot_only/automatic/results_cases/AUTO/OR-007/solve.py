import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    paths = ['Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv', 'Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv', 'Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv']
    df_demand = pd.read_csv(paths[0], dtype=str, keep_default_na=False)
    if 'customer' not in df_demand.columns or 'demand' not in df_demand.columns:
        raise ValueError("customer_demand.csv must contain 'customer' and 'demand' columns")
    df_demand['demand'] = df_demand['demand'].astype(float)
    demand = dict(zip(df_demand['customer'], df_demand['demand']))
    df_supply = pd.read_csv(paths[1], dtype=str, keep_default_na=False)
    if 'region' not in df_supply.columns or 'supply_capacity' not in df_supply.columns:
        raise ValueError("supply_capacity.csv must contain 'region' and 'supply_capacity' columns")
    df_supply['supply_capacity'] = df_supply['supply_capacity'].astype(float)
    supply = dict(zip(df_supply['region'], df_supply['supply_capacity']))
    df_cost = pd.read_csv(paths[2], dtype=str, keep_default_na=False)
    if df_cost.shape[1] < 2:
        raise ValueError('transportation_costs.csv must have at least two columns')
    warehouse_col = df_cost.columns[0]
    store_cols = df_cost.columns[1:]
    for col in store_cols:
        df_cost[col] = df_cost[col].astype(float)
    cost = {}
    for (_, row) in df_cost.iterrows():
        warehouse = row[warehouse_col]
        for store in store_cols:
            cost[warehouse, store] = row[store]
    I = list(supply.keys())
    J = list(demand.keys())
    for i in I:
        for j in J:
            if (i, j) not in cost:
                raise ValueError(f'Missing cost entry for warehouse {i} and store {j}')
    m = gurobipy.Model('GreenMart_TP')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in I for j in J], lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((cost[i, j] * x[i, j] for i in I for j in J)), GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[i, j] for i in I)) >= demand[j] for j in J), name='')
    m.addConstrs((gurobipy.quicksum((x[i, j] for j in J)) <= supply[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()