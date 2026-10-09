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
    customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/customer_demand.csv'
    supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/supply_capacity.csv'
    transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/transportation_costs.csv'
    customer_df = read_csv_with_encodings(customer_demand_path)
    supply_df = read_csv_with_encodings(supply_capacity_path)
    cost_df = read_csv_with_encodings(transportation_costs_path)
    suppliers = supply_df['Unnamed: 0'].tolist()
    customers = customer_df['customer'].tolist()
    try:
        demand = {row['customer']: float(row['demand']) for (_, row) in customer_df.iterrows()}
    except Exception as e:
        raise ValueError(f'Error parsing demand: {e}')
    try:
        supply_capacity = {row['Unnamed: 0']: float(row['supply_capacity']) for (_, row) in supply_df.iterrows()}
    except Exception as e:
        raise ValueError(f'Error parsing supply_capacity: {e}')
    cost = {}
    for (_, row) in cost_df.iterrows():
        supplier = row['Unnamed: 0']
        if supplier not in suppliers:
            continue
        cost[supplier] = {}
        for customer in customers:
            if customer not in cost_df.columns:
                raise ValueError(f'Customer {customer} not found in cost matrix columns.')
            try:
                cost_val = float(row[customer])
            except Exception as e:
                raise ValueError(f'Error parsing cost for supplier {supplier}, customer {customer}: {e}')
            cost[supplier][customer] = cost_val
    for i in suppliers:
        if i not in cost:
            raise ValueError(f'Missing cost row for supplier {i}')
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Missing cost entry for supplier {i}, customer {j}')
    for j in customers:
        if j not in demand:
            raise ValueError(f'Missing demand for customer {j}')
    for i in suppliers:
        if i not in supply_capacity:
            raise ValueError(f'Missing supply_capacity for supplier {i}')
    m = gp.Model('TP3_Original_RAG')
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