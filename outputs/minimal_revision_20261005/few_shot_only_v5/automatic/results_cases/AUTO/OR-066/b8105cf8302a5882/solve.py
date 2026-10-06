import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_try_encodings(path, **kwargs):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with supported encodings.')
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv'
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv'
    trans_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv'
    demand_df = read_csv_try_encodings(demand_path)
    if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
        raise ValueError("demand.csv must have columns 'customer' and 'demand'")
    customers = demand_df['customer'].astype(str).tolist()
    demand = demand_df.groupby('customer')['demand'].sum().to_dict()
    fixed_df = read_csv_try_encodings(fixed_cost_path)
    if 'Unnamed: 0' in fixed_df.columns:
        supplier_col = 'Unnamed: 0'
    elif 'supplier' in fixed_df.columns:
        supplier_col = 'supplier'
    else:
        raise ValueError('fixed_cost.csv must have a supplier column')
    if 'fixed_costs' not in fixed_df.columns:
        raise ValueError("fixed_cost.csv must have column 'fixed_costs'")
    suppliers = fixed_df[supplier_col].astype(str).tolist()
    fixed_cost = fixed_df.set_index(supplier_col)['fixed_costs'].to_dict()
    trans_df = read_csv_try_encodings(trans_cost_path)
    if 'Unnamed: 0' in trans_df.columns:
        trans_supplier_col = 'Unnamed: 0'
    elif 'supplier' in trans_df.columns:
        trans_supplier_col = 'supplier'
    else:
        raise ValueError('transportation_costs.csv must have a supplier column')
    cost_customer_cols = [col for col in trans_df.columns if col != trans_supplier_col]
    missing_customers = set(customers) - set(cost_customer_cols)
    if missing_customers:
        raise ValueError(f'transportation_costs.csv missing columns for customers: {missing_customers}')
    missing_suppliers = set(suppliers) - set(trans_df[trans_supplier_col].astype(str))
    if missing_suppliers:
        raise ValueError(f'transportation_costs.csv missing rows for suppliers: {missing_suppliers}')
    cost = {}
    for (_, row) in trans_df.iterrows():
        s = str(row[trans_supplier_col])
        if s not in suppliers:
            continue
        cost[s] = {}
        for c in customers:
            cost[s][c] = float(row[c])
    for s in suppliers:
        if s not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {s}')
        if s not in cost:
            raise ValueError(f'Missing transportation cost row for supplier {s}')
        for c in customers:
            if c not in cost[s]:
                raise ValueError(f'Missing transportation cost for supplier {s}, customer {c}')
    for c in customers:
        if c not in demand:
            raise ValueError(f'Missing demand for customer {c}')
    M = sum(demand.values())
    m = gp.Model('UFLP')
    m.Params.MIPGap = 0.0001
    x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[s][c] * x[s, c] for s in suppliers for c in customers)) + gp.quicksum((fixed_cost[s] * y[s] for s in suppliers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[s, c] for s in suppliers)) == demand[c] for c in customers), name='')
    m.addConstrs((gp.quicksum((x[s, c] for c in customers)) <= M * y[s] for s in suppliers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()