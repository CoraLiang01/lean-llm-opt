import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/customer_demand.csv'
    supply_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/supply_capacity.csv'
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/transportation_costs.csv'

    def read_csv_try_encodings(path):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    demand_df = read_csv_try_encodings(demand_path)
    if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
        raise ValueError("customer_demand.csv must have columns 'customer' and 'demand'")
    customers = demand_df['customer'].tolist()
    demand = {}
    for (_, row) in demand_df.iterrows():
        cust = row['customer']
        try:
            val = float(row['demand'])
        except Exception:
            raise ValueError(f"Non-numeric demand for customer {cust}: {row['demand']}")
        if cust in demand:
            demand[cust] += val
        else:
            demand[cust] = val
    supply_df = read_csv_try_encodings(supply_path)
    supplier_col = None
    for col in supply_df.columns:
        if col.casefold() in ['supplier', 'unnamed: 0']:
            supplier_col = col
            break
    if supplier_col is None or 'supply_capacity' not in supply_df.columns:
        raise ValueError("supply_capacity.csv must have a supplier column and 'supply_capacity'")
    suppliers = supply_df[supplier_col].tolist()
    supply_capacity = {}
    for (_, row) in supply_df.iterrows():
        sup = row[supplier_col]
        try:
            val = float(row['supply_capacity'])
        except Exception:
            raise ValueError(f"Non-numeric supply_capacity for supplier {sup}: {row['supply_capacity']}")
        if sup in supply_capacity:
            supply_capacity[sup] += val
        else:
            supply_capacity[sup] = val
    cost_df = read_csv_try_encodings(cost_path)
    cost_supplier_col = None
    for col in cost_df.columns:
        if col.casefold() in ['supplier', 'unnamed: 0']:
            cost_supplier_col = col
            break
    if cost_supplier_col is None:
        raise ValueError('transportation_costs.csv must have a supplier row index column')
    cost_customer_cols = [col for col in cost_df.columns if col != cost_supplier_col]
    missing_customers = [j for j in customers if j not in cost_customer_cols]
    if missing_customers:
        raise ValueError(f'Customers {missing_customers} in demand not found in transportation_costs.csv columns')
    cost_suppliers = cost_df[cost_supplier_col].tolist()
    missing_suppliers = [i for i in suppliers if i not in cost_suppliers]
    if missing_suppliers:
        raise ValueError(f'Suppliers {missing_suppliers} in supply not found in transportation_costs.csv rows')
    cost = {}
    for (_, row) in cost_df.iterrows():
        sup = row[cost_supplier_col]
        if sup not in suppliers:
            continue
        cost[sup] = {}
        for cust in customers:
            val = row[cust]
            try:
                cost[sup][cust] = float(val)
            except Exception:
                raise ValueError(f'Non-numeric cost for supplier {sup}, customer {cust}: {val}')
    for i in suppliers:
        if i not in cost:
            raise ValueError(f'Missing cost row for supplier {i}')
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Missing cost coefficient for supplier {i}, customer {j}')
    quantity_keys = [(i, j) for i in suppliers for j in customers]
    m = gp.Model('TP3')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for (i, j) in quantity_keys)), GRB.MINIMIZE)
    for j in customers:
        m.addConstr(gp.quicksum((quantity_vars[i, j] for i in suppliers)) >= demand[j], name=f'demand_{j}')
    for i in suppliers:
        m.addConstr(gp.quicksum((quantity_vars[i, j] for j in customers)) <= supply_capacity[i], name=f'supply_{i}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()