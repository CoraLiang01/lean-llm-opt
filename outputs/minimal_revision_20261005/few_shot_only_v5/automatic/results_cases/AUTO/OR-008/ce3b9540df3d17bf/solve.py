import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_robust(path, **kwargs):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/customer_demand.csv'
    supply_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/supply_capacity.csv'
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv'
    demand_df = read_csv_robust(demand_path)
    supply_df = read_csv_robust(supply_path)
    cost_df = read_csv_robust(cost_path)
    demand_df['Customers'] = demand_df['Customers'].astype(str).str.strip()
    supply_df['Suppliers'] = supply_df['Suppliers'].astype(str).str.strip()
    cost_df.rename(columns={cost_df.columns[0]: 'Suppliers'}, inplace=True)
    cost_df['Suppliers'] = cost_df['Suppliers'].astype(str).str.strip()
    suppliers = supply_df['Suppliers'].unique().tolist()
    customers = demand_df['Customers'].unique().tolist()
    demand = {}
    for (_, row) in demand_df.iterrows():
        key = str(row['Customers']).strip()
        if key in demand:
            demand[key] += float(row['demand'])
        else:
            demand[key] = float(row['demand'])
    supply_capacity = {}
    for (_, row) in supply_df.iterrows():
        key = str(row['Suppliers']).strip()
        if key in supply_capacity:
            supply_capacity[key] += float(row['supply_capacity'])
        else:
            supply_capacity[key] = float(row['supply_capacity'])
    cost = {s: {} for s in suppliers}
    for (_, row) in cost_df.iterrows():
        s = str(row['Suppliers']).strip()
        for c in customers:
            if c not in row:
                raise ValueError(f"Customer '{c}' not found in transportation_costs.csv columns.")
            cost[s][c] = float(row[c])
    for c in customers:
        if c not in demand:
            raise ValueError(f"Demand for customer '{c}' missing.")
    for s in suppliers:
        if s not in supply_capacity:
            raise ValueError(f"Supply capacity for supplier '{s}' missing.")
    for s in suppliers:
        for c in customers:
            if c not in cost[s]:
                raise ValueError(f"Cost for supplier '{s}' to customer '{c}' missing.")
    m = gp.Model('FreshMart_Transportation')
    m.Params.MIPGap = 0.0001
    keys = [(s, c) for s in suppliers for c in customers]
    x = m.addVars(keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[s][c] * x[s, c] for (s, c) in keys)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[s, c] for s in suppliers)) >= demand[c] for c in customers), name='')
    m.addConstrs((gp.quicksum((x[s, c] for c in customers)) <= supply_capacity[s] for s in suppliers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()