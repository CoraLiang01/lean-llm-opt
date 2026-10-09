import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_with_encodings(path, **kwargs):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode file: {path}')
    customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/customer_demand.csv'
    supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/supply_capacity.csv'
    transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv'
    customer_df = read_csv_with_encodings(customer_demand_path, dtype=str, keep_default_na=False)
    if 'Customers' not in customer_df.columns or 'demand' not in customer_df.columns:
        raise ValueError("customer_demand.csv must have columns 'Customers' and 'demand'")
    customers = customer_df['Customers'].tolist()
    demand = {}
    for (_, row) in customer_df.iterrows():
        cust = row['Customers']
        try:
            val = float(row['demand'])
        except Exception:
            raise ValueError(f"Non-numeric demand for customer {cust}: {row['demand']}")
        demand[cust] = val
    supply_df = read_csv_with_encodings(supply_capacity_path, dtype=str, keep_default_na=False)
    if 'Suppliers' not in supply_df.columns or 'supply_capacity' not in supply_df.columns:
        raise ValueError("supply_capacity.csv must have columns 'Suppliers' and 'supply_capacity'")
    suppliers = supply_df['Suppliers'].tolist()
    supply_capacity = {}
    for (_, row) in supply_df.iterrows():
        sup = row['Suppliers']
        try:
            val = float(row['supply_capacity'])
        except Exception:
            raise ValueError(f"Non-numeric supply_capacity for supplier {sup}: {row['supply_capacity']}")
        supply_capacity[sup] = val
    cost_df = read_csv_with_encodings(transportation_costs_path, dtype=str, keep_default_na=False)
    if 'Unnamed: 0' not in cost_df.columns:
        raise ValueError("transportation_costs.csv must have row identifier column 'Unnamed: 0'")
    cost_suppliers = cost_df['Unnamed: 0'].tolist()
    cost_customers = [col for col in cost_df.columns if col != 'Unnamed: 0']
    missing_suppliers = set(suppliers) - set(cost_suppliers)
    missing_customers = set(customers) - set(cost_customers)
    if missing_suppliers:
        raise ValueError(f'Missing suppliers in transportation_costs.csv: {missing_suppliers}')
    if missing_customers:
        raise ValueError(f'Missing customers in transportation_costs.csv: {missing_customers}')
    cost = {}
    for (_, row) in cost_df.iterrows():
        sup = row['Unnamed: 0']
        if sup not in suppliers:
            continue
        cost[sup] = {}
        for cust in customers:
            val = row.get(cust, '')
            try:
                cost[sup][cust] = float(val)
            except Exception:
                raise ValueError(f'Non-numeric cost for supplier {sup}, customer {cust}: {val}')
    for i in suppliers:
        if i not in cost:
            raise ValueError(f'Missing cost row for supplier {i}')
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Missing cost entry for supplier {i}, customer {j}')
    m = gp.Model('FreshMart_Transportation')
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