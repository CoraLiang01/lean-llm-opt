import pandas as pd
import gurobipy
from gurobipy import GRB
customer_demand_path = 'Test_Dataset/Large-scale-or/TP_testing/TP4/customer_demand.csv'
supply_capacity_path = 'Test_Dataset/Large-scale-or/TP_testing/TP4/supply_capacity.csv'
transportation_costs_path = 'Test_Dataset/Large-scale-or/TP_testing/TP4/transportation_costs.csv'

def read_customer_demand(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if 'customer' not in df.columns or 'demand' not in df.columns:
        raise ValueError("customer_demand.csv must contain 'customer' and 'demand' columns")
    demand = {}
    for (_, row) in df.iterrows():
        cust = row['customer'].strip()
        val = row['demand'].strip()
        if cust == '' or val == '':
            raise ValueError(f'Missing demand value for customer {cust}')
        try:
            demand[cust] = float(val)
        except ValueError:
            raise ValueError(f'Invalid numeric demand for customer {cust}: {val}')
    return demand

def read_supply_capacity(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if 'Unnamed: 0' not in df.columns or 'supply_capacity' not in df.columns:
        raise ValueError("supply_capacity.csv must contain 'Unnamed: 0' and 'supply_capacity' columns")
    supply = {}
    for (_, row) in df.iterrows():
        center = row['Unnamed: 0'].strip()
        val = row['supply_capacity'].strip()
        if center == '' or val == '':
            raise ValueError(f'Missing supply value for center {center}')
        try:
            supply[center] = float(val)
        except ValueError:
            raise ValueError(f'Invalid numeric supply for center {center}: {val}')
    return supply

def read_transportation_costs(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if 'Unnamed: 0' not in df.columns:
        raise ValueError("transportation_costs.csv must contain 'Unnamed: 0' column")
    cost = {}
    customer_cols = [col for col in df.columns if col != 'Unnamed: 0']
    for (_, row) in df.iterrows():
        supplier = row['Unnamed: 0'].strip()
        cost[supplier] = {}
        for cust in customer_cols:
            val = row[cust].strip()
            if val == '':
                raise ValueError(f'Missing cost for supplier {supplier} to customer {cust}')
            try:
                cost[supplier][cust] = float(val)
            except ValueError:
                raise ValueError(f'Invalid numeric cost for supplier {supplier} to customer {cust}: {val}')
    return cost

def solve_problem():
    demand = read_customer_demand(customer_demand_path)
    supply = read_supply_capacity(supply_capacity_path)
    cost = read_transportation_costs(transportation_costs_path)
    I = set(supply.keys())
    J = set(demand.keys())
    for i in I:
        if i not in cost:
            raise ValueError(f'Missing cost data for supplier {i}')
        for j in J:
            if j not in cost[i]:
                raise ValueError(f'Missing cost data for supplier {i} to customer {j}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in I for j in J], lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((cost[i][j] * x[i, j] for i in I for j in J)), GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[i, j] for i in I)) >= demand[j] for j in J), name='')
    m.addConstrs((gurobipy.quicksum((x[i, j] for j in J)) <= supply[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()