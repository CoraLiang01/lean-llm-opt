import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csvs = [('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/customer_demand.csv', ['Customers', 'demand']), ('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/supply_capacity.csv', ['Suppliers', 'supply_capacity']), ('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv', None)]

    def read_csv(path, **kwargs):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with supported encodings.')
    df_demand = read_csv(csvs[0][0])
    if not set(['Customers', 'demand']).issubset(df_demand.columns):
        raise ValueError('customer_demand.csv missing required columns.')
    customers = df_demand['Customers'].tolist()
    demand = dict(zip(df_demand['Customers'], df_demand['demand']))
    df_supply = read_csv(csvs[1][0])
    if not set(['Suppliers', 'supply_capacity']).issubset(df_supply.columns):
        raise ValueError('supply_capacity.csv missing required columns.')
    suppliers = df_supply['Suppliers'].tolist()
    supply_capacity = dict(zip(df_supply['Suppliers'], df_supply['supply_capacity']))
    df_cost = read_csv(csvs[2][0])
    if df_cost.columns[0].casefold() not in {'suppliers', 'supplier', 'unnamed: 0'}:
        raise ValueError('First column of transportation_costs.csv must be supplier names.')
    cost_suppliers = df_cost.iloc[:, 0].tolist()
    cost_customers = df_cost.columns[1:].tolist()
    if [s.casefold() for s in cost_suppliers] != [s.casefold() for s in suppliers]:
        raise ValueError('Supplier order mismatch between supply_capacity.csv and transportation_costs.csv.')
    if [c.casefold() for c in cost_customers] != [c.casefold() for c in customers]:
        raise ValueError('Customer order mismatch between customer_demand.csv and transportation_costs.csv.')
    cost = {}
    for (i, s) in enumerate(suppliers):
        row = df_cost.iloc[i, 1:].tolist()
        if len(row) != len(customers):
            raise ValueError(f'Row {i} in transportation_costs.csv does not match number of customers.')
        cost[s] = dict(zip(customers, row))
    for s in suppliers:
        if s not in supply_capacity:
            raise ValueError(f'Supplier {s} missing in supply_capacity.')
        if s not in cost:
            raise ValueError(f'Supplier {s} missing in cost matrix.')
        for c in customers:
            if c not in cost[s]:
                raise ValueError(f'Customer {c} missing in cost matrix for supplier {s}.')
    for c in customers:
        if c not in demand:
            raise ValueError(f'Customer {c} missing in demand.')
    m = gp.Model('FreshMart_TP')
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