import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_with_encodings(path):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/customer_demand.csv'
    supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/supply_capacity.csv'
    transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/transportation_costs.csv'
    customer_demand_df = read_csv_with_encodings(customer_demand_path)
    supply_capacity_df = read_csv_with_encodings(supply_capacity_path)
    transportation_costs_df = read_csv_with_encodings(transportation_costs_path)
    suppliers = supply_capacity_df['Unnamed: 0'].tolist()
    customers = customer_demand_df['customer'].tolist()
    try:
        demand = {row['customer']: float(row['demand']) for (_, row) in customer_demand_df.iterrows()}
    except Exception as e:
        raise RuntimeError(f'Error parsing demand: {e}')
    try:
        supply_capacity = {row['Unnamed: 0']: float(row['supply_capacity']) for (_, row) in supply_capacity_df.iterrows()}
    except Exception as e:
        raise RuntimeError(f'Error parsing supply_capacity: {e}')
    cost = {}
    for (_, row) in transportation_costs_df.iterrows():
        supplier = row['Unnamed: 0']
        if supplier not in suppliers:
            continue
        cost[supplier] = {}
        for customer in customers:
            if customer not in row:
                match_cols = [col for col in transportation_costs_df.columns if col.casefold() == customer.casefold()]
                if match_cols:
                    col = match_cols[0]
                else:
                    raise RuntimeError(f"Customer '{customer}' not found as column in transportation_costs.csv")
            else:
                col = customer
            try:
                val = row[col]
                cost[supplier][customer] = float(val)
            except Exception as e:
                raise RuntimeError(f"Error parsing cost for supplier '{supplier}', customer '{customer}': {e}")
    for i in suppliers:
        if i not in cost:
            raise RuntimeError(f"Missing cost row for supplier '{i}'")
        for j in customers:
            if j not in cost[i]:
                raise RuntimeError(f"Missing cost for supplier '{i}', customer '{j}'")
    for j in customers:
        if j not in demand:
            raise RuntimeError(f"Missing demand for customer '{j}'")
    for i in suppliers:
        if i not in supply_capacity:
            raise RuntimeError(f"Missing supply_capacity for supplier '{i}'")
    m = gp.Model('Walmart_Transportation')
    m.Params.MIPGap = 0.0001
    quantity_keys = [(i, j) for i in suppliers for j in customers]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for i in suppliers for j in customers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for i in suppliers)) >= demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for j in customers)) <= supply_capacity[i] for i in suppliers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()