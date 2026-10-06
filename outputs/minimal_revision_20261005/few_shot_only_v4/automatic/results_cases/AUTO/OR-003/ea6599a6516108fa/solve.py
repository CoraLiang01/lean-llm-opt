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
    customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/customer_demand.csv'
    supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/supply_capacity.csv'
    transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/transportation_costs.csv'
    df_demand = read_csv_with_encodings(customer_demand_path)
    df_supply = read_csv_with_encodings(supply_capacity_path)
    df_cost = read_csv_with_encodings(transportation_costs_path)
    if 'customer' in df_demand.columns:
        customers = df_demand['customer'].astype(str).tolist()
        demand = dict(zip(df_demand['customer'].astype(str), df_demand['demand']))
    else:
        raise ValueError("customer_demand.csv must have a 'customer' column.")
    supplier_col = df_supply.columns[0]
    if 'supply_capacity' in df_supply.columns:
        suppliers = df_supply[supplier_col].astype(str).tolist()
        supply_capacity = dict(zip(df_supply[supplier_col].astype(str), df_supply['supply_capacity']))
    else:
        raise ValueError("supply_capacity.csv must have a 'supply_capacity' column.")
    cost_supplier_col = df_cost.columns[0]
    cost_customer_cols = [col for col in df_cost.columns if col != cost_supplier_col]
    missing_customers = set(customers) - set(cost_customer_cols)
    if missing_customers:
        raise ValueError(f'Missing customers in transportation_costs.csv: {missing_customers}')
    missing_suppliers = set(suppliers) - set(df_cost[cost_supplier_col].astype(str))
    if missing_suppliers:
        raise ValueError(f'Missing suppliers in transportation_costs.csv: {missing_suppliers}')
    cost = {}
    for (_, row) in df_cost.iterrows():
        supplier = str(row[cost_supplier_col])
        if supplier not in suppliers:
            continue
        cost[supplier] = {}
        for customer in customers:
            cost[supplier][customer] = row[customer]
    for i in suppliers:
        if i not in cost:
            raise ValueError(f'Missing cost row for supplier {i}')
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Missing cost for supplier {i}, customer {j}')
    m = gp.Model('TP3')
    m.Params.MIPGap = 0.0001
    x_keys = [(i, j) for i in suppliers for j in customers]
    x = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in customers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) >= demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= supply_capacity[i] for i in suppliers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()