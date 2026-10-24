import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    demand_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/demand.csv'
    fixed_cost_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/fixed_cost.csv'
    transport_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv'
    demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
    fixed_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
    trans_df = pd.read_csv(transport_path, dtype=str, keep_default_na=False)
    if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
        raise ValueError('Demand CSV must contain columns "customer" and "demand"')
    customers = demand_df['customer'].tolist()
    d = {}
    for (_, row) in demand_df.iterrows():
        cust = row['customer']
        try:
            val = float(row['demand'])
        except ValueError:
            raise ValueError(f'Invalid demand value for customer {cust}')
        d[cust] = val
    if 'Unnamed: 0' not in fixed_df.columns or 'fixed_costs' not in fixed_df.columns:
        raise ValueError('Fixed cost CSV must contain columns "Unnamed: 0" and "fixed_costs"')
    warehouses = fixed_df['Unnamed: 0'].tolist()
    f = {}
    for (_, row) in fixed_df.iterrows():
        wh = row['Unnamed: 0']
        try:
            val = float(row['fixed_costs'])
        except ValueError:
            raise ValueError(f'Invalid fixed cost value for warehouse {wh}')
        f[wh] = val
    if 'Unnamed: 0' not in trans_df.columns:
        raise ValueError('Transportation costs CSV must contain column "Unnamed: 0"')
    trans_columns = [col for col in trans_df.columns if col != 'Unnamed: 0']
    missing_cols = set(customers) - set(trans_columns)
    if missing_cols:
        raise ValueError(f'Transportation costs missing columns for customers: {missing_cols}')
    c = {}
    for (_, row) in trans_df.iterrows():
        wh = row['Unnamed: 0']
        cost_dict = {}
        for cust in customers:
            try:
                val = float(row[cust])
            except ValueError:
                raise ValueError(f'Invalid transportation cost for warehouse {wh} to customer {cust}')
            cost_dict[cust] = val
        c[wh] = cost_dict
    if set(warehouses) != set(f.keys()):
        missing = set(f.keys()) - set(warehouses)
        raise ValueError(f'Warehouse identifiers mismatch: missing in fixed cost data: {missing}')
    if set(warehouses) != set(c.keys()):
        missing = set(c.keys()) - set(warehouses)
        raise ValueError(f'Warehouse identifiers mismatch: missing in transportation data: {missing}')
    for wh in warehouses:
        if set(c[wh].keys()) != set(customers):
            missing = set(customers) - set(c[wh].keys())
            raise ValueError(f'Warehouse {wh} missing transportation costs for customers: {missing}')
    if set(customers) != set(d.keys()):
        missing = set(d.keys()) - set(customers)
        raise ValueError(f'Customer identifiers mismatch: missing in demand data: {missing}')
    M = sum(d.values())
    m = gurobipy.Model('UFLP7')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in warehouses for j in customers], lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(warehouses, vtype=GRB.BINARY, name='')
    transport_obj = gurobipy.quicksum((c[i][j] * x[i, j] for i in warehouses for j in customers))
    fixed_obj = gurobipy.quicksum((f[i] * y[i] for i in warehouses))
    m.setObjective(transport_obj + fixed_obj, GRB.MINIMIZE)
    for j in customers:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in warehouses)) == d[j], name=f'demand_{j}')
    for i in warehouses:
        m.addConstr(gurobipy.quicksum((x[i, j] for j in customers)) <= M * y[i], name=f'activation_{i}')
    m.optimize()
    return m
m = solve_problem()