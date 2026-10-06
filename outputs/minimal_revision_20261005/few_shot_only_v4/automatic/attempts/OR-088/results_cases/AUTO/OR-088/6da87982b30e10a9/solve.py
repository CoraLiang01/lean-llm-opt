import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/cost.csv'
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/demand.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            cost_df = pd.read_csv(cost_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {cost_path} with tried encodings.')
    for enc in encodings:
        try:
            demand_df = pd.read_csv(demand_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {demand_path} with tried encodings.')
    if 'plant' not in cost_df.columns:
        raise ValueError("Missing 'plant' column in cost.csv")
    plants = list(cost_df['plant'])
    customer_cols = [col for col in cost_df.columns if col.startswith('C') and col[1:].isdigit()]
    customers = customer_cols
    if len(customers) == 0:
        raise ValueError('No customer columns (C1..C15) found in cost.csv')
    if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
        raise ValueError("Missing 'customer' or 'demand' column in demand.csv")
    demand_df['customer'] = demand_df['customer'].astype(str)
    demand = {}
    for (_, row) in demand_df.iterrows():
        cust = row['customer']
        if cust not in customers:
            continue
        demand[cust] = float(row['demand'])
    missing_customers = set(customers) - set(demand.keys())
    if missing_customers:
        raise ValueError(f'Demand missing for customers: {missing_customers}')
    if 'fixed_cost' not in cost_df.columns or 'capacity' not in cost_df.columns:
        raise ValueError("Missing 'fixed_cost' or 'capacity' column in cost.csv")
    fixed_cost = {}
    capacity = {}
    for (_, row) in cost_df.iterrows():
        plant = row['plant']
        fixed_cost[plant] = float(row['fixed_cost'])
        capacity[plant] = float(row['capacity'])
    cost = {}
    for (_, row) in cost_df.iterrows():
        plant = row['plant']
        cost[plant] = {}
        for cust in customers:
            if cust not in row:
                raise ValueError(f'Missing cost column {cust} for plant {plant}')
            cost[plant][cust] = float(row[cust])
    for plant in plants:
        if plant not in fixed_cost or plant not in capacity or plant not in cost:
            raise ValueError(f'Missing data for plant {plant}')
        for cust in customers:
            if cust not in cost[plant]:
                raise ValueError(f'Missing cost for plant {plant}, customer {cust}')
    for cust in customers:
        if cust not in demand:
            raise ValueError(f'Missing demand for customer {cust}')
    m = gp.Model('UFLP')
    m.Params.MIPGap = 0.0001
    x_keys = [(i, j) for i in plants for j in customers]
    y_keys = plants
    x = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(y_keys, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in plants)) + gp.quicksum((cost[i][j] * x[i, j] for i in plants for j in customers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in plants)) == demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= capacity[i] * y[i] for i in plants), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()