import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_robust(path, **kwargs):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(path, encoding=enc, dtype=str, keep_default_na=False, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with supported encodings.')
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/demand.csv'
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/fixed_cost.csv'
    transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/transportation_costs.csv'
    demand_df = read_csv_robust(demand_path)
    if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
        raise ValueError("demand.csv must have columns 'customer' and 'demand'")
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
    fixed_cost_df = read_csv_robust(fixed_cost_path)
    if 'Unnamed: 0' in fixed_cost_df.columns:
        supplier_col = 'Unnamed: 0'
    elif 'supplier' in fixed_cost_df.columns:
        supplier_col = 'supplier'
    else:
        raise ValueError('fixed_cost.csv must have a supplier index column')
    if 'fixed_costs' not in fixed_cost_df.columns:
        raise ValueError("fixed_cost.csv must have column 'fixed_costs'")
    suppliers = fixed_cost_df[supplier_col].tolist()
    fixed_cost = {}
    for (_, row) in fixed_cost_df.iterrows():
        sup = row[supplier_col]
        try:
            val = float(row['fixed_costs'])
        except Exception:
            raise ValueError(f"Non-numeric fixed cost for supplier {sup}: {row['fixed_costs']}")
        if sup in fixed_cost:
            raise ValueError(f'Duplicate supplier {sup} in fixed_cost.csv')
        fixed_cost[sup] = val
    trans_df = read_csv_robust(transportation_costs_path)
    if 'Unnamed: 0' in trans_df.columns:
        trans_supplier_col = 'Unnamed: 0'
    elif 'supplier' in trans_df.columns:
        trans_supplier_col = 'supplier'
    else:
        raise ValueError('transportation_costs.csv must have a supplier index column')
    cost_customer_cols = [col for col in trans_df.columns if col != trans_supplier_col]
    missing_customers = set(customers) - set(cost_customer_cols)
    if missing_customers:
        raise ValueError(f'Missing transportation cost columns for customers: {missing_customers}')
    cost = {}
    for (_, row) in trans_df.iterrows():
        sup = row[trans_supplier_col]
        if sup not in suppliers:
            continue
        if sup in cost:
            raise ValueError(f'Duplicate supplier {sup} in transportation_costs.csv')
        cost[sup] = {}
        for cust in customers:
            try:
                val = float(row[cust])
            except Exception:
                raise ValueError(f'Non-numeric transportation cost for supplier {sup}, customer {cust}: {row[cust]}')
            cost[sup][cust] = val
    for sup in suppliers:
        if sup not in cost:
            raise ValueError(f'Supplier {sup} missing in transportation_costs.csv')
        for cust in customers:
            if cust not in cost[sup]:
                raise ValueError(f'Supplier {sup}, customer {cust} missing in transportation_costs.csv')
    for cust in customers:
        if cust not in demand:
            raise ValueError(f'Customer {cust} missing in demand.csv')
    x_keys = [(i, j) for i in suppliers for j in customers]
    y_keys = suppliers
    m = gp.Model('UFLP')
    quantity_vars = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    open_vars = m.addVars(y_keys, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for (i, j) in x_keys)) + gp.quicksum((fixed_cost[i] * open_vars[i] for i in y_keys)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for i in suppliers)) == demand[j] for j in customers), name='')
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