import pandas as pd
import gurobipy
from gurobipy import GRB
customer_path = 'Test_Dataset/Large-scale-or/TP_testing/TP8/customer_demand.csv'
supply_path = 'Test_Dataset/Large-scale-or/TP_testing/TP8/supply_capacity.csv'
cost_path = 'Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv'
customer_df = pd.read_csv(customer_path, dtype=str, keep_default_na=False)
supply_df = pd.read_csv(supply_path, dtype=str, keep_default_na=False)
cost_df = pd.read_csv(cost_path, dtype=str, keep_default_na=False)
suppliers = ['Supplier1', 'Supplier2', 'Supplier3', 'Supplier4', 'Supplier5']
customers = ['Customer1', 'Customer2', 'Customer3', 'Customer4', 'Customer5', 'Customer6']
demand = {}
for (_, row) in customer_df.iterrows():
    cust = row['Customers']
    if cust not in customers:
        continue
    val = row['demand']
    try:
        demand[cust] = float(val)
    except ValueError:
        raise ValueError(f'Invalid demand value for {cust}: {val}')
for cust in customers:
    if cust not in demand:
        raise ValueError(f'Missing demand for {cust}')
supply = {}
for (_, row) in supply_df.iterrows():
    sup = row['Suppliers']
    if sup not in suppliers:
        continue
    val = row['supply_capacity']
    try:
        supply[sup] = float(val)
    except ValueError:
        raise ValueError(f'Invalid supply value for {sup}: {val}')
for sup in suppliers:
    if sup not in supply:
        raise ValueError(f'Missing supply for {sup}')
cost = {}
for (_, row) in cost_df.iterrows():
    sup = row['Unnamed: 0']
    if sup not in suppliers:
        continue
    cost[sup] = {}
    for cust in customers:
        val = row[cust]
        try:
            cost[sup][cust] = float(val)
        except ValueError:
            raise ValueError(f'Invalid cost for {sup}-{cust}: {val}')
for sup in suppliers:
    if sup not in cost:
        raise ValueError(f'Missing cost row for {sup}')
    for cust in customers:
        if cust not in cost[sup]:
            raise ValueError(f'Missing cost for {sup}-{cust}')

def solve_problem():
    m = gurobipy.Model('Transportation')
    m.setParam('MIPGap', 1e-06)
    keys = [(i, j) for i in suppliers for j in customers]
    x = m.addVars(keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in customers)), GRB.MINIMIZE)
    for j in customers:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in suppliers)) >= demand[j], name=f'demand_{j}')
    for i in suppliers:
        m.addConstr(gurobipy.quicksum((x[i, j] for j in customers)) <= supply[i], name=f'supply_{i}')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')