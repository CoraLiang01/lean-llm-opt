import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv(path, **kwargs):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, dtype=str, keep_default_na=False, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/customer_demand.csv'
    supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/supply_capacity.csv'
    transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/transportation_costs.csv'
    df_demand = read_csv(customer_demand_path)
    df_supply = read_csv(supply_capacity_path)
    df_cost = read_csv(transportation_costs_path)
    suppliers = df_supply['Supplier'].tolist()
    customers = df_demand['Customers'].tolist()
    try:
        demand = {row['Customers']: float(row['demand']) for (_, row) in df_demand.iterrows()}
    except Exception as e:
        raise ValueError(f'Error parsing demand values: {e}')
    try:
        supply_capacity = {row['Supplier']: float(row['supply_capacity']) for (_, row) in df_supply.iterrows()}
    except Exception as e:
        raise ValueError(f'Error parsing supply capacity values: {e}')
    if 'Unnamed: 0' not in df_cost.columns:
        raise ValueError("transportation_costs.csv must have 'Unnamed: 0' as supplier row index.")
    df_cost = df_cost.set_index('Unnamed: 0')
    missing_suppliers = set(suppliers) - set(df_cost.index)
    missing_customers = set(customers) - set(df_cost.columns)
    if missing_suppliers:
        raise ValueError(f'Missing suppliers in transportation_costs.csv: {missing_suppliers}')
    if missing_customers:
        raise ValueError(f'Missing customers in transportation_costs.csv: {missing_customers}')
    cost = {}
    for i in suppliers:
        cost[i] = {}
        for j in customers:
            val = df_cost.at[i, j]
            try:
                cost[i][j] = float(val)
            except Exception as e:
                raise ValueError(f'Invalid cost value for supplier {i}, customer {j}: {val}')
    for j in customers:
        if j not in demand:
            raise ValueError(f'Customer {j} missing in demand data.')
    for i in suppliers:
        if i not in supply_capacity:
            raise ValueError(f'Supplier {i} missing in supply capacity data.')
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Cost missing for supplier {i}, customer {j}.')
    m = gp.Model('Amazon_Distribution_Transportation')
    m.Params.MIPGap = 0.0001
    quantity_keys = [(i, j) for i in suppliers for j in customers]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for i in suppliers for j in customers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for i in suppliers)) >= demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for j in customers)) <= supply_capacity[i] for i in suppliers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()