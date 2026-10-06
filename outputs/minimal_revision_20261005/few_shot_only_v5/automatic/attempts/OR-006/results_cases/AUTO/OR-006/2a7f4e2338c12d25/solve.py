import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/customer_demand.csv'
    supply_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/supply_capacity.csv'
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/transportation_costs.csv'

    def read_csv_with_encodings(path, **kwargs):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    demand_df = read_csv_with_encodings(demand_path)
    if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
        raise ValueError("customer_demand.csv must have columns ['customer', 'demand']")
    demand_df['customer'] = demand_df['customer'].astype(str)
    customers = demand_df['customer'].tolist()
    demand = dict(zip(demand_df['customer'], demand_df['demand']))
    supply_df = read_csv_with_encodings(supply_path)
    if 'Unnamed: 0' in supply_df.columns:
        supply_df['warehouse'] = supply_df['Unnamed: 0'].astype(str)
    elif 'warehouse' in supply_df.columns:
        supply_df['warehouse'] = supply_df['warehouse'].astype(str)
    else:
        raise ValueError('supply_capacity.csv must have a warehouse identifier column')
    if 'supply_capacity' not in supply_df.columns:
        raise ValueError("supply_capacity.csv must have column 'supply_capacity'")
    warehouses = supply_df['warehouse'].tolist()
    supply_capacity = dict(zip(supply_df['warehouse'], supply_df['supply_capacity']))
    cost_df = read_csv_with_encodings(cost_path)
    if 'Unnamed: 0' in cost_df.columns:
        cost_df['warehouse'] = cost_df['Unnamed: 0'].astype(str)
    elif 'warehouse' in cost_df.columns:
        cost_df['warehouse'] = cost_df['warehouse'].astype(str)
    else:
        raise ValueError('transportation_costs.csv must have a warehouse identifier column')
    cost_df = cost_df.set_index('warehouse')
    missing_warehouses = set(warehouses) - set(cost_df.index)
    missing_customers = set(customers) - set(cost_df.columns)
    if missing_warehouses:
        raise ValueError(f'Missing warehouses in transportation_costs.csv: {missing_warehouses}')
    if missing_customers:
        raise ValueError(f'Missing customers in transportation_costs.csv: {missing_customers}')
    cost = {i: {j: float(cost_df.loc[i, j]) for j in customers} for i in warehouses}
    for j in customers:
        if j not in demand:
            raise ValueError(f'Customer {j} missing in demand data')
    for i in warehouses:
        if i not in supply_capacity:
            raise ValueError(f'Warehouse {i} missing in supply data')
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Cost missing for warehouse {i}, customer {j}')
    m = gp.Model('TP6')
    m.Params.MIPGap = 0.0001
    x_keys = [(i, j) for i in warehouses for j in customers]
    x = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in warehouses for j in customers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in warehouses)) >= demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= supply_capacity[i] for i in warehouses), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()