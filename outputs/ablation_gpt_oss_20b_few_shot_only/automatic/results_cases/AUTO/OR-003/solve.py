import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    path_demand = 'Test_Dataset/Large-scale-or/TP_testing/TP3/customer_demand.csv'
    path_supply = 'Test_Dataset/Large-scale-or/TP_testing/TP3/supply_capacity.csv'
    path_cost = 'Test_Dataset/Large-scale-or/TP_testing/TP3/transportation_costs.csv'
    df_demand = pd.read_csv(path_demand, dtype=str, keep_default_na=False)
    if 'Unnamed: 0' not in df_demand.columns or 'demand' not in df_demand.columns:
        raise ValueError('Customer demand CSV missing required columns.')
    demand = {}
    for (_, row) in df_demand.iterrows():
        cust = row['Unnamed: 0']
        val = row['demand']
        if val == '':
            raise ValueError(f'Missing demand value for customer {cust}.')
        demand[cust] = float(val)
    df_supply = pd.read_csv(path_supply, dtype=str, keep_default_na=False)
    if 'Unnamed: 0' not in df_supply.columns or 'supply_capacity' not in df_supply.columns:
        raise ValueError('Supply capacity CSV missing required columns.')
    supply = {}
    for (_, row) in df_supply.iterrows():
        sup = row['Unnamed: 0']
        val = row['supply_capacity']
        if val == '':
            raise ValueError(f'Missing supply capacity for supplier {sup}.')
        supply[sup] = float(val)
    df_cost = pd.read_csv(path_cost, dtype=str, keep_default_na=False)
    if 'Unnamed: 0' not in df_cost.columns:
        raise ValueError('Transportation costs CSV missing supplier identifier column.')
    cost = {}
    customer_columns = [col for col in df_cost.columns if col != 'Unnamed: 0']
    for (_, row) in df_cost.iterrows():
        sup = row['Unnamed: 0']
        cost[sup] = {}
        for cust in customer_columns:
            val = row[cust]
            if val == '':
                raise ValueError(f'Missing cost for supplier {sup} to customer {cust}.')
            cost[sup][cust] = float(val)
    suppliers = set(supply.keys())
    if suppliers != set(cost.keys()):
        missing_suppliers = suppliers.symmetric_difference(set(cost.keys()))
        raise ValueError(f'Supplier identifiers mismatch: {missing_suppliers}')
    customers = set(demand.keys())
    if customers != set(customer_columns):
        missing_customers = customers.symmetric_difference(set(customer_columns))
        raise ValueError(f'Customer identifiers mismatch: {missing_customers}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    pairs = [(i, j) for i in suppliers for j in customers]
    x = m.addVars(pairs, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in customers)), GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[i, j] for i in suppliers)) >= demand[j] for j in customers), name='')
    m.addConstrs((gurobipy.quicksum((x[i, j] for j in customers)) <= supply[i] for i in suppliers), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')