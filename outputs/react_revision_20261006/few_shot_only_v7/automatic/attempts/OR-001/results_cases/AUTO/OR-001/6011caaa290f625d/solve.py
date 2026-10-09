import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    try_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']

    def read_csv_with_encodings(path):
        for enc in try_encodings:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/customer_demand.csv'
    supply_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/supply_capacity.csv'
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/transportation_costs.csv'
    demand_df = read_csv_with_encodings(demand_path)
    supply_df = read_csv_with_encodings(supply_path)
    cost_df = read_csv_with_encodings(cost_path)
    if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
        raise ValueError("customer_demand.csv must have columns 'customer' and 'demand'")
    customers = demand_df['customer'].tolist()
    if 'Unnamed: 0' not in supply_df.columns or 'supply_capacity' not in supply_df.columns:
        raise ValueError("supply_capacity.csv must have columns 'Unnamed: 0' and 'supply_capacity'")
    suppliers = supply_df['Unnamed: 0'].tolist()
    if 'Unnamed: 0' not in cost_df.columns:
        raise ValueError("transportation_costs.csv must have column 'Unnamed: 0' for supplier names")
    cost_suppliers = cost_df['Unnamed: 0'].tolist()
    cost_customers = [col for col in cost_df.columns if col != 'Unnamed: 0']
    missing_suppliers = set(suppliers) - set(cost_suppliers)
    if missing_suppliers:
        raise ValueError(f'Missing suppliers in transportation_costs.csv: {missing_suppliers}')
    missing_customers = set(customers) - set(cost_customers)
    if missing_customers:
        raise ValueError(f'Missing customers in transportation_costs.csv: {missing_customers}')
    try:
        demand = {row['customer']: float(row['demand']) for (_, row) in demand_df.iterrows()}
    except Exception as e:
        raise ValueError(f'Error converting demand values: {e}')
    try:
        supply_capacity = {row['Unnamed: 0']: float(row['supply_capacity']) for (_, row) in supply_df.iterrows()}
    except Exception as e:
        raise ValueError(f'Error converting supply_capacity values: {e}')
    cost = {}
    for (_, row) in cost_df.iterrows():
        supplier = row['Unnamed: 0']
        cost[supplier] = {}
        for customer in customers:
            val = row[customer]
            try:
                cost[supplier][customer] = float(val)
            except Exception as e:
                raise ValueError(f'Error converting cost for ({supplier},{customer}): {e}')
    for i in suppliers:
        if i not in cost:
            raise ValueError(f'Supplier {i} missing in cost matrix.')
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Cost entry missing for ({i},{j}) in cost matrix.')
    m = gp.Model('Amazon_Distribution_Transportation')
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