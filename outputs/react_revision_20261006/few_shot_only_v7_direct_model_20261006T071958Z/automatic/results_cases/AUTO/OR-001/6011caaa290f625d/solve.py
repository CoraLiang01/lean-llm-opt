import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_robust(path, **kwargs):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/customer_demand.csv'
    supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/supply_capacity.csv'
    transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/transportation_costs.csv'
    customer_df = read_csv_robust(customer_demand_path, dtype=str, keep_default_na=False)
    supply_df = read_csv_robust(supply_capacity_path, dtype=str, keep_default_na=False)
    cost_df = read_csv_robust(transportation_costs_path, dtype=str, keep_default_na=False)
    I = supply_df['Unnamed: 0'].tolist()
    J = customer_df['customer'].tolist()
    try:
        demand = {row['customer']: float(row['demand']) for (_, row) in customer_df.iterrows()}
    except Exception as e:
        raise ValueError(f'Error parsing demand values: {e}')
    try:
        supply_capacity = {row['Unnamed: 0']: float(row['supply_capacity']) for (_, row) in supply_df.iterrows()}
    except Exception as e:
        raise ValueError(f'Error parsing supply_capacity values: {e}')
    cost = {}
    for (_, row) in cost_df.iterrows():
        supplier = row['Unnamed: 0']
        if supplier not in I:
            continue
        cost[supplier] = {}
        for customer in J:
            if customer not in cost_df.columns:
                raise ValueError(f"Customer '{customer}' not found as column in transportation_costs.csv")
            val = row[customer]
            try:
                cost[supplier][customer] = float(val)
            except Exception as e:
                raise ValueError(f"Error parsing cost for supplier '{supplier}', customer '{customer}': {e}")
    for i in I:
        if i not in cost:
            raise ValueError(f"Missing cost row for supplier '{i}'")
        for j in J:
            if j not in cost[i]:
                raise ValueError(f"Missing cost entry for supplier '{i}', customer '{j}'")
    for j in J:
        if j not in demand:
            raise ValueError(f"Missing demand for customer '{j}'")
    for i in I:
        if i not in supply_capacity:
            raise ValueError(f"Missing supply_capacity for supplier '{i}'")
    m = gp.Model('Amazon_Distribution_Transportation')
    m.Params.MIPGap = 0.0001
    quantity_keys = [(i, j) for i in I for j in J]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for i in I for j in J)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for i in I)) >= demand[j] for j in J), name='')
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for j in J)) <= supply_capacity[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()