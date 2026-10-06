import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def read_csv_with_encodings(path, **kwargs):
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f'Could not decode {path} with tried encodings.')

def solve_problem():
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/demand.csv'
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/fixed_cost.csv'
    trans_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/transportation_costs.csv'
    demand_df = read_csv_with_encodings(demand_path)
    if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
        raise ValueError("demand.csv must have columns 'customer' and 'demand'")
    customers = demand_df['customer'].astype(str).tolist()
    demand = {}
    for (_, row) in demand_df.iterrows():
        cust = str(row['customer'])
        if cust in demand:
            demand[cust] += float(row['demand'])
        else:
            demand[cust] = float(row['demand'])
    fixed_df = read_csv_with_encodings(fixed_cost_path)
    if 'Unnamed: 0' in fixed_df.columns:
        supplier_col = 'Unnamed: 0'
    elif 'supplier' in fixed_df.columns:
        supplier_col = 'supplier'
    else:
        raise ValueError('fixed_cost.csv must have a supplier index column')
    if 'fixed_costs' not in fixed_df.columns:
        raise ValueError("fixed_cost.csv must have column 'fixed_costs'")
    suppliers = fixed_df[supplier_col].astype(str).tolist()
    fixed_cost = {}
    for (_, row) in fixed_df.iterrows():
        sup = str(row[supplier_col])
        if sup in fixed_cost:
            fixed_cost[sup] += float(row['fixed_costs'])
        else:
            fixed_cost[sup] = float(row['fixed_costs'])
    trans_df = read_csv_with_encodings(trans_cost_path)
    if 'Unnamed: 0' in trans_df.columns:
        trans_supplier_col = 'Unnamed: 0'
    elif 'supplier' in trans_df.columns:
        trans_supplier_col = 'supplier'
    else:
        raise ValueError('transportation_costs.csv must have a supplier index column')
    trans_customers = [col for col in trans_df.columns if col != trans_supplier_col]
    missing_customers = set(customers) - set(trans_customers)
    if missing_customers:
        raise ValueError(f'Customers {missing_customers} in demand.csv not found in transportation_costs.csv')
    trans_suppliers = trans_df[trans_supplier_col].astype(str).tolist()
    missing_suppliers = set(suppliers) - set(trans_suppliers)
    if missing_suppliers:
        raise ValueError(f'Suppliers {missing_suppliers} in fixed_cost.csv not found in transportation_costs.csv')
    cost = {}
    for (_, row) in trans_df.iterrows():
        sup = str(row[trans_supplier_col])
        if sup not in suppliers:
            continue
        cost[sup] = {}
        for cust in customers:
            if cust not in row:
                raise ValueError(f'Customer {cust} not found in transportation_costs.csv for supplier {sup}')
            cost[sup][cust] = float(row[cust])
    I = suppliers
    J = customers
    for i in I:
        if i not in cost:
            raise ValueError(f'Missing cost row for supplier {i}')
        for j in J:
            if j not in cost[i]:
                raise ValueError(f'Missing cost entry for supplier {i}, customer {j}')
    for i in I:
        if i not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {i}')
    for j in J:
        if j not in demand:
            raise ValueError(f'Missing demand for customer {j}')
    M = sum((demand[j] for j in J))
    m = gp.Model('UFLP3')
    m.Params.MIPGap = 0.0001
    x = m.addVars(I, J, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in I)) + gp.quicksum((cost[i][j] * x[i, j] for i in I for j in J)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in I)) == demand[j] for j in J), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in J)) <= M * y[i] for i in I), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')