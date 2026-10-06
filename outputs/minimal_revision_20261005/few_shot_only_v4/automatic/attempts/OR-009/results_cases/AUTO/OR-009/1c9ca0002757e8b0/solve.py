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
    customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/customer_demand.csv'
    supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/supply_capacity.csv'
    transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv'
    demand_df = read_csv_with_encodings(customer_demand_path)
    supply_df = read_csv_with_encodings(supply_capacity_path)
    cost_df = read_csv_with_encodings(transportation_costs_path)
    demand_df.columns = [c.strip() for c in demand_df.columns]
    supply_df.columns = [c.strip() for c in supply_df.columns]
    cost_df.columns = [c.strip() for c in cost_df.columns]
    plants = [str(x) for x in supply_df['Unnamed: 0']]
    outlets = [str(x) for x in demand_df['customer']]
    demand = {}
    for (_, row) in demand_df.iterrows():
        cust = str(row['customer'])
        if cust in demand:
            demand[cust] += float(row['demand'])
        else:
            demand[cust] = float(row['demand'])
    supply_capacity = {}
    for (_, row) in supply_df.iterrows():
        plant = str(row['Unnamed: 0'])
        if plant in supply_capacity:
            supply_capacity[plant] += float(row['supply_capacity'])
        else:
            supply_capacity[plant] = float(row['supply_capacity'])
    cost = {}
    for (_, row) in cost_df.iterrows():
        plant = str(row['Unnamed: 0'])
        cost[plant] = {}
        for outlet in outlets:
            if outlet not in cost_df.columns:
                raise ValueError(f'Outlet {outlet} not found in transportation_costs.csv columns.')
            val = row[outlet]
            try:
                cost[plant][outlet] = float(val)
            except Exception:
                raise ValueError(f'Invalid cost value for ({plant}, {outlet}): {val}')
    for plant in plants:
        if plant not in cost:
            raise ValueError(f'Missing cost row for plant {plant}')
        for outlet in outlets:
            if outlet not in cost[plant]:
                raise ValueError(f'Missing cost for ({plant}, {outlet})')
    for plant in plants:
        if plant not in supply_capacity:
            raise ValueError(f'Missing supply capacity for plant {plant}')
    for outlet in outlets:
        if outlet not in demand:
            raise ValueError(f'Missing demand for outlet {outlet}')
    m = gp.Model('BrewCo_Transportation')
    m.Params.MIPGap = 0.0001
    keys = [(i, j) for i in plants for j in outlets]
    x = m.addVars(keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in plants for j in outlets)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in plants)) >= demand[j] for j in outlets), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in outlets)) <= supply_capacity[i] for i in plants), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()