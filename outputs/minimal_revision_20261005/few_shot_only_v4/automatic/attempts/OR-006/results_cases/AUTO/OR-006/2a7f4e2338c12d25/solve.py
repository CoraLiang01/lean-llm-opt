import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csvs = [('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/customer_demand.csv', ['customer', 'demand']), ('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/supply_capacity.csv', ['Unnamed: 0', 'supply_capacity']), ('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/transportation_costs.csv', ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10'])]

    def read_csv(path, **kwargs):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with supported encodings.')
    df_demand = read_csv(csvs[0][0])
    if not set(['customer', 'demand']).issubset(df_demand.columns):
        raise ValueError('customer_demand.csv missing required columns.')
    customers = df_demand['customer'].astype(str).tolist()
    demand = dict(zip(df_demand['customer'].astype(str), df_demand['demand']))
    df_supply = read_csv(csvs[1][0])
    if not set(['Unnamed: 0', 'supply_capacity']).issubset(df_supply.columns):
        raise ValueError('supply_capacity.csv missing required columns.')
    suppliers = df_supply['Unnamed: 0'].astype(str).tolist()
    supply_capacity = dict(zip(df_supply['Unnamed: 0'].astype(str), df_supply['supply_capacity']))
    df_cost = read_csv(csvs[2][0])
    if 'Unnamed: 0' not in df_cost.columns:
        raise ValueError("transportation_costs.csv missing 'Unnamed: 0' column.")
    cost_suppliers = df_cost['Unnamed: 0'].astype(str).tolist()
    cost_customers = [col for col in df_cost.columns if col != 'Unnamed: 0']
    if set(suppliers) != set(cost_suppliers):
        raise ValueError('Mismatch between supply_capacity and transportation_costs warehouse indices.')
    if set(customers) != set(cost_customers):
        raise ValueError('Mismatch between customer_demand and transportation_costs customer indices.')
    cost = {}
    for (i, row) in df_cost.iterrows():
        supplier = str(row['Unnamed: 0'])
        cost[supplier] = {}
        for customer in customers:
            if customer not in row:
                raise ValueError(f'Customer {customer} missing in transportation_costs.csv for supplier {supplier}.')
            cost[supplier][customer] = float(row[customer])
    for i in suppliers:
        if i not in cost:
            raise ValueError(f'Missing cost row for supplier {i}.')
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Missing cost for ({i},{j}).')
    for j in customers:
        if j not in demand:
            raise ValueError(f'Missing demand for customer {j}.')
    for i in suppliers:
        if i not in supply_capacity:
            raise ValueError(f'Missing supply capacity for supplier {i}.')
    m = gp.Model('TP6')
    m.Params.MIPGap = 0.0001
    keys = [(i, j) for i in suppliers for j in customers]
    x = m.addVars(keys, lb=0, vtype=GRB.CONTINUOUS, name='')
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