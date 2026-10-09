import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_paths = {'customer_demand': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/customer_demand.csv', 'supply_capacity': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/supply_capacity.csv', 'transportation_costs': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/transportation_costs.csv'}
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']

    def read_csv_with_encodings(path):
        for enc in encodings:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    df_demand = read_csv_with_encodings(csv_paths['customer_demand'])
    if 'customer' not in df_demand.columns or 'demand' not in df_demand.columns:
        raise ValueError("customer_demand.csv must have columns 'customer' and 'demand'")
    customers = df_demand['customer'].tolist()
    demand = {}
    for (_, row) in df_demand.iterrows():
        cust = row['customer']
        try:
            demand_val = float(row['demand'])
        except Exception:
            raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
        demand[cust] = demand_val
    df_supply = read_csv_with_encodings(csv_paths['supply_capacity'])
    if 'Unnamed: 0' in df_supply.columns:
        supplier_col = 'Unnamed: 0'
    elif 'supplier' in df_supply.columns:
        supplier_col = 'supplier'
    else:
        raise ValueError('supply_capacity.csv must have a supplier identifier column')
    if 'supply_capacity' not in df_supply.columns:
        raise ValueError("supply_capacity.csv must have column 'supply_capacity'")
    suppliers = df_supply[supplier_col].tolist()
    supply_capacity = {}
    for (_, row) in df_supply.iterrows():
        sup = row[supplier_col]
        try:
            supply_val = float(row['supply_capacity'])
        except Exception:
            raise ValueError(f"Invalid supply_capacity value for supplier {sup}: {row['supply_capacity']}")
        supply_capacity[sup] = supply_val
    df_cost = read_csv_with_encodings(csv_paths['transportation_costs'])
    if 'Unnamed: 0' in df_cost.columns:
        cost_supplier_col = 'Unnamed: 0'
    elif 'supplier' in df_cost.columns:
        cost_supplier_col = 'supplier'
    else:
        raise ValueError('transportation_costs.csv must have a supplier identifier column')
    cost_customer_cols = [col for col in df_cost.columns if col != cost_supplier_col]
    missing_customers = set(customers) - set(cost_customer_cols)
    if missing_customers:
        raise ValueError(f'Missing cost columns for customers: {missing_customers}')
    missing_suppliers = set(suppliers) - set(df_cost[cost_supplier_col])
    if missing_suppliers:
        raise ValueError(f'Missing cost rows for suppliers: {missing_suppliers}')
    cost = {}
    for (_, row) in df_cost.iterrows():
        sup = row[cost_supplier_col]
        if sup not in suppliers:
            continue
        cost[sup] = {}
        for cust in customers:
            try:
                cost_val = float(row[cust])
            except Exception:
                raise ValueError(f'Invalid cost value for supplier {sup}, customer {cust}: {row[cust]}')
            cost[sup][cust] = cost_val
    for sup in suppliers:
        if sup not in cost:
            raise ValueError(f'Missing cost row for supplier {sup}')
        for cust in customers:
            if cust not in cost[sup]:
                raise ValueError(f'Missing cost entry for supplier {sup}, customer {cust}')
    m = gp.Model('TP4_Optimal_Fulfillment')
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