import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    cost_csv_paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/cost.csv']
    demand_csv_paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/demand.csv']

    def read_csv_with_encodings(path):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    cost_df = read_csv_with_encodings(cost_csv_paths[0])
    demand_df = read_csv_with_encodings(demand_csv_paths[0])
    plant_col = 'plant'
    fixed_cost_col = 'fixed_cost'
    capacity_col = 'capacity'
    customer_cols = [col for col in cost_df.columns if col.startswith('C') and col[1:].isdigit()]
    plants = cost_df[plant_col].tolist()
    customers = customer_cols
    for col in [plant_col, fixed_cost_col, capacity_col] + customer_cols:
        if col not in cost_df.columns:
            raise ValueError(f'Missing column {col} in cost.csv')
    if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
        raise ValueError('Missing required columns in demand.csv')
    fixed_cost = {}
    capacity = {}
    cost = {}
    for (idx, row) in cost_df.iterrows():
        plant = row[plant_col]
        try:
            fixed_cost[plant] = float(row[fixed_cost_col])
        except Exception:
            raise ValueError(f'Invalid fixed_cost for plant {plant}: {row[fixed_cost_col]}')
        try:
            capacity[plant] = float(row[capacity_col])
        except Exception:
            raise ValueError(f'Invalid capacity for plant {plant}: {row[capacity_col]}')
        cost[plant] = {}
        for cust in customer_cols:
            try:
                cost[plant][cust] = float(row[cust])
            except Exception:
                raise ValueError(f'Invalid cost for plant {plant}, customer {cust}: {row[cust]}')
    demand = {}
    for (idx, row) in demand_df.iterrows():
        cust = row['customer']
        if cust not in customers:
            continue
        try:
            demand[cust] = float(row['demand'])
        except Exception:
            raise ValueError(f"Invalid demand for customer {cust}: {row['demand']}")
    missing_customers = [c for c in customers if c not in demand]
    if missing_customers:
        raise ValueError(f'Missing demand for customers: {missing_customers}')
    m = gp.Model('UFLP')
    m.Params.MIPGap = 0.0001
    quantity_keys = [(i, j) for i in plants for j in customers]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    open_vars = m.addVars(plants, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((fixed_cost[i] * open_vars[i] for i in plants)) + gp.quicksum((cost[i][j] * quantity_vars[i, j] for (i, j) in quantity_keys)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for i in plants)) == demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for j in customers)) <= capacity[i] for i in plants), name='')
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for j in customers)) <= capacity[i] * open_vars[i] for i in plants), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()