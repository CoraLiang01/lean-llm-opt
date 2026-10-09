import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_paths = {'customer_demand': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/customer_demand.csv', 'supply_capacity': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/supply_capacity.csv', 'transportation_costs': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/transportation_costs.csv'}
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
        raise ValueError('customer_demand.csv must have columns: customer, demand')
    customers = df_demand['customer'].tolist()
    demand = {}
    for (_, row) in df_demand.iterrows():
        cust = row['customer']
        try:
            val = float(row['demand'])
        except Exception:
            raise ValueError(f"Non-numeric demand for customer {cust}: {row['demand']}")
        if cust in demand:
            demand[cust] += val
        else:
            demand[cust] = val
    df_supply = read_csv_with_encodings(csv_paths['supply_capacity'])
    store_col = 'Unnamed: 0' if 'Unnamed: 0' in df_supply.columns else 'store'
    if store_col not in df_supply.columns or 'supply_capacity' not in df_supply.columns:
        raise ValueError('supply_capacity.csv must have columns: Unnamed: 0 (or store), supply_capacity')
    suppliers = df_supply[store_col].tolist()
    supply_capacity = {}
    for (_, row) in df_supply.iterrows():
        store = row[store_col]
        try:
            val = float(row['supply_capacity'])
        except Exception:
            raise ValueError(f"Non-numeric supply_capacity for store {store}: {row['supply_capacity']}")
        if store in supply_capacity:
            supply_capacity[store] += val
        else:
            supply_capacity[store] = val
    df_cost = read_csv_with_encodings(csv_paths['transportation_costs'])
    cost_store_col = 'Unnamed: 0' if 'Unnamed: 0' in df_cost.columns else 'store'
    if cost_store_col not in df_cost.columns:
        raise ValueError('transportation_costs.csv must have a store identifier column')
    cost_customer_cols = [col for col in df_cost.columns if col != cost_store_col]
    missing_customers = set(customers) - set(cost_customer_cols)
    if missing_customers:
        raise ValueError(f'Customers missing in transportation_costs.csv: {missing_customers}')
    missing_suppliers = set(suppliers) - set(df_cost[cost_store_col])
    if missing_suppliers:
        raise ValueError(f'Suppliers missing in transportation_costs.csv: {missing_suppliers}')
    cost = {}
    for (_, row) in df_cost.iterrows():
        store = row[cost_store_col]
        if store not in suppliers:
            continue
        cost[store] = {}
        for cust in customers:
            val = row[cust]
            try:
                costval = float(val)
            except Exception:
                raise ValueError(f'Non-numeric cost for store {store}, customer {cust}: {val}')
            cost[store][cust] = costval
    for i in suppliers:
        if i not in cost:
            raise ValueError(f'Missing cost row for supplier {i}')
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Missing cost for supplier {i}, customer {j}')
    m = gp.Model('Walmart_Transportation')
    quantity_keys = [(i, j) for i in suppliers for j in customers]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for (i, j) in quantity_keys)), GRB.MINIMIZE)
    for j in customers:
        m.addConstr(gp.quicksum((quantity_vars[i, j] for i in suppliers)) >= demand[j], name=f'demand_{j}')
    for i in suppliers:
        m.addConstr(gp.quicksum((quantity_vars[i, j] for j in customers)) <= supply_capacity[i], name=f'supply_{i}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()