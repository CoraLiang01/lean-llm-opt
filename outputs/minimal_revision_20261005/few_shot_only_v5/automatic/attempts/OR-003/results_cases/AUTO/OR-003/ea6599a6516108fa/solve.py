import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/customer_demand.csv'
    supply_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/supply_capacity.csv'
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/transportation_costs.csv'

    def read_csv_robust(path, **kwargs):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    df_demand = read_csv_robust(demand_path)
    if 'customer' not in df_demand.columns or 'demand' not in df_demand.columns:
        raise ValueError("customer_demand.csv must have columns ['customer', 'demand']")
    customers = df_demand['customer'].astype(str).tolist()
    demand = df_demand.set_index('customer')['demand'].to_dict()
    df_supply = read_csv_robust(supply_path)
    supplier_col = 'Unnamed: 0' if 'Unnamed: 0' in df_supply.columns else 'supplier' if 'supplier' in df_supply.columns else None
    if supplier_col is None or 'supply_capacity' not in df_supply.columns:
        raise ValueError("supply_capacity.csv must have columns ['Unnamed: 0' or 'supplier', 'supply_capacity']")
    suppliers = df_supply[supplier_col].astype(str).tolist()
    supply_capacity = df_supply.set_index(supplier_col)['supply_capacity'].to_dict()
    df_cost = read_csv_robust(cost_path)
    cost_supplier_col = 'Unnamed: 0' if 'Unnamed: 0' in df_cost.columns else 'supplier' if 'supplier' in df_cost.columns else None
    if cost_supplier_col is None:
        raise ValueError('transportation_costs.csv must have a supplier row index column')
    missing_customers = [c for c in customers if c not in df_cost.columns]
    if missing_customers:
        raise ValueError(f'transportation_costs.csv missing columns for customers: {missing_customers}')
    missing_suppliers = [s for s in suppliers if s not in df_cost[cost_supplier_col].astype(str).tolist()]
    if missing_suppliers:
        raise ValueError(f'transportation_costs.csv missing rows for suppliers: {missing_suppliers}')
    df_cost = df_cost.set_index(cost_supplier_col)
    cost = {}
    for i in suppliers:
        if i not in df_cost.index:
            raise ValueError(f'Supplier {i} missing in transportation_costs.csv')
        cost[i] = {}
        for j in customers:
            val = df_cost.at[i, j]
            if pd.isnull(val):
                raise ValueError(f'Missing cost for supplier {i}, customer {j}')
            cost[i][j] = float(val)
    if set(demand.keys()) != set(customers):
        raise ValueError('Mismatch in customer keys between demand and customer list')
    if set(supply_capacity.keys()) != set(suppliers):
        raise ValueError('Mismatch in supplier keys between supply_capacity and supplier list')
    for i in suppliers:
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Cost missing for supplier {i}, customer {j}')
    m = gp.Model('TP3')
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