import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    path_demand = 'Test_Dataset/Large-scale-or/TP_testing/TP6/customer_demand.csv'
    path_supply = 'Test_Dataset/Large-scale-or/TP_testing/TP6/supply_capacity.csv'
    path_cost = 'Test_Dataset/Large-scale-or/TP_testing/TP6/transportation_costs.csv'
    df_demand = pd.read_csv(path_demand, dtype=str, keep_default_na=False)
    if df_demand.shape[1] < 2:
        raise ValueError('Customer demand file must have at least two columns.')
    store_col = df_demand.columns[0]
    demand_col = df_demand.columns[1]
    demand = {}
    for (_, row) in df_demand.iterrows():
        store = row[store_col].strip()
        val = row[demand_col].strip()
        if val == '':
            raise ValueError(f'Missing demand value for store {store}')
        try:
            demand[store] = float(val)
        except ValueError:
            raise ValueError(f"Invalid demand value '{val}' for store {store}")
    df_supply = pd.read_csv(path_supply, dtype=str, keep_default_na=False)
    if df_supply.shape[1] < 2:
        raise ValueError('Supply capacity file must have at least two columns.')
    warehouse_col = df_supply.columns[0]
    supply_col = df_supply.columns[1]
    supply = {}
    for (_, row) in df_supply.iterrows():
        warehouse = row[warehouse_col].strip()
        val = row[supply_col].strip()
        if val == '':
            raise ValueError(f'Missing supply value for warehouse {warehouse}')
        try:
            supply[warehouse] = float(val)
        except ValueError:
            raise ValueError(f"Invalid supply value '{val}' for warehouse {warehouse}")
    df_cost = pd.read_csv(path_cost, dtype=str, keep_default_na=False)
    if df_cost.shape[1] < 2:
        raise ValueError('Transportation costs file must have at least two columns.')
    cost_warehouse_col = df_cost.columns[0]
    cost_store_cols = df_cost.columns[1:]
    cost = {}
    for (_, row) in df_cost.iterrows():
        warehouse = row[cost_warehouse_col].strip()
        cost[warehouse] = {}
        for store in cost_store_cols:
            val = row[store].strip()
            if val == '':
                raise ValueError(f'Missing cost value for warehouse {warehouse} to store {store}')
            try:
                cost[warehouse][store] = float(val)
            except ValueError:
                raise ValueError(f"Invalid cost value '{val}' for warehouse {warehouse} to store {store}")
    warehouses = set(supply.keys())
    stores = set(demand.keys())
    if not warehouses.issubset(cost.keys()):
        missing = warehouses - set(cost.keys())
        raise ValueError(f'Missing cost entries for warehouses: {missing}')
    for w in warehouses:
        if set(cost[w].keys()) != stores:
            missing = stores - set(cost[w].keys())
            raise ValueError(f'Missing cost entries for warehouse {w} to stores: {missing}')
    m = gurobipy.Model('TransportationProblem')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in warehouses for j in stores], lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((cost[i][j] * x[i, j] for i in warehouses for j in stores)), GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[i, j] for i in warehouses)) >= demand[j] for j in stores), name='')
    m.addConstrs((gurobipy.quicksum((x[i, j] for j in stores)) <= supply[i] for i in warehouses), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')