import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/customer_demand.csv'
    supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/supply_capacity.csv'
    transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/transportation_costs.csv'

    def read_csv_with_encodings(path, **kwargs):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not read {path} with tried encodings.')
    df_demand = read_csv_with_encodings(customer_demand_path)
    if 'customer' not in df_demand.columns or 'demand' not in df_demand.columns:
        raise ValueError("customer_demand.csv must have columns 'customer' and 'demand'")
    customers = df_demand['customer'].astype(str).tolist()
    demand = df_demand.set_index('customer')['demand'].to_dict()
    df_supply = read_csv_with_encodings(supply_capacity_path)
    if 'Unnamed: 0' not in df_supply.columns or 'supply_capacity' not in df_supply.columns:
        raise ValueError("supply_capacity.csv must have columns 'Unnamed: 0' and 'supply_capacity'")
    suppliers = df_supply['Unnamed: 0'].astype(str).tolist()
    supply_capacity = df_supply.set_index('Unnamed: 0')['supply_capacity'].to_dict()
    df_cost = read_csv_with_encodings(transportation_costs_path)
    if 'Unnamed: 0' not in df_cost.columns:
        raise ValueError("transportation_costs.csv must have column 'Unnamed: 0' for suppliers")
    cost_customer_cols = [col for col in df_cost.columns if col != 'Unnamed: 0']
    missing_customers = set(customers) - set(cost_customer_cols)
    if missing_customers:
        raise ValueError(f'Missing cost columns for customers: {missing_customers}')
    missing_suppliers = set(suppliers) - set(df_cost['Unnamed: 0'].astype(str))
    if missing_suppliers:
        raise ValueError(f'Missing cost rows for suppliers: {missing_suppliers}')
    cost = {}
    for (_, row) in df_cost.iterrows():
        i = str(row['Unnamed: 0'])
        cost[i] = {}
        for j in customers:
            if j not in row:
                raise ValueError(f'Cost for supplier {i} to customer {j} missing in transportation_costs.csv')
            cost[i][j] = float(row[j])
    for i in suppliers:
        if i not in cost:
            raise ValueError(f'Supplier {i} missing in cost data')
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Cost for supplier {i} to customer {j} missing')
    for j in customers:
        if j not in demand:
            raise ValueError(f'Demand for customer {j} missing')
    for i in suppliers:
        if i not in supply_capacity:
            raise ValueError(f'Supply capacity for supplier {i} missing')
    m = gp.Model('TP4')
    m.Params.MIPGap = 0.0001
    x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in customers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) >= demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= supply_capacity[i] for i in suppliers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()