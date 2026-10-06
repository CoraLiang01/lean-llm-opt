import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_with_encodings(path):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError(f'Could not decode {path} with tried encodings.')
    customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/customer_demand.csv'
    supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/supply_capacity.csv'
    transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/transportation_costs.csv'
    customer_demand_df = read_csv_with_encodings(customer_demand_path)
    supply_capacity_df = read_csv_with_encodings(supply_capacity_path)
    transportation_costs_df = read_csv_with_encodings(transportation_costs_path)
    suppliers = list(supply_capacity_df['Supplier'])
    customers = list(customer_demand_df['Customers'])
    demand = dict(zip(customer_demand_df['Customers'], customer_demand_df['demand']))
    supply_capacity = dict(zip(supply_capacity_df['Supplier'], supply_capacity_df['supply_capacity']))
    cost = {}
    cost_suppliers = list(transportation_costs_df.iloc[:, 0])
    cost_customers = list(transportation_costs_df.columns[1:])
    if cost_suppliers != suppliers:
        raise ValueError('Supplier order in transportation_costs.csv does not match supply_capacity.csv')
    if cost_customers != customers:
        raise ValueError('Customer order in transportation_costs.csv does not match customer_demand.csv')
    for (i, supplier) in enumerate(suppliers):
        row = transportation_costs_df.iloc[i, 1:]
        cost[supplier] = {}
        for (j, customer) in enumerate(customers):
            val = row.iloc[j]
            if pd.isnull(val):
                raise ValueError(f'Missing cost for ({supplier}, {customer})')
            cost[supplier][customer] = float(val)
    m = gp.Model('TP5')
    m.Params.MIPGap = 0.0001
    x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
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