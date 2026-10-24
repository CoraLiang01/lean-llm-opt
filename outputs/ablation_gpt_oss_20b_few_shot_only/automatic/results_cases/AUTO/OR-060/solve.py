import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    demand_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/demand.csv'
    fixed_cost_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/fixed_cost.csv'
    cost_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/transportation_costs.csv'
    df_demand = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
    df_fixed = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
    df_cost = pd.read_csv(cost_path, dtype=str, keep_default_na=False)
    df_demand['customer'] = df_demand['customer'].str.strip()
    df_fixed['Unnamed: 0'] = df_fixed['Unnamed: 0'].str.strip()
    df_cost['Unnamed: 0'] = df_cost['Unnamed: 0'].str.strip()
    df_demand['demand'] = df_demand['demand'].astype(float)
    df_fixed['fixed_costs'] = df_fixed['fixed_costs'].astype(float)
    cost_cols = [c for c in df_cost.columns if c != 'Unnamed: 0']
    df_cost[cost_cols] = df_cost[cost_cols].astype(float)
    suppliers = df_fixed['Unnamed: 0'].tolist()
    customers = df_demand['customer'].tolist()
    fixed_cost = dict(zip(df_fixed['Unnamed: 0'], df_fixed['fixed_costs']))
    demand = dict(zip(df_demand['customer'], df_demand['demand']))
    cost = {}
    for (_, row) in df_cost.iterrows():
        supplier = row['Unnamed: 0']
        if supplier not in suppliers:
            raise ValueError(f'Supplier {supplier} in cost matrix not in fixed cost list')
        cost[supplier] = {}
        for cust in customers:
            if cust not in row:
                raise ValueError(f'Customer {cust} missing in cost row for supplier {supplier}')
            cost[supplier][cust] = float(row[cust])
    if set(cost.keys()) != set(suppliers):
        missing = set(suppliers) - set(cost.keys())
        raise ValueError(f'Missing cost data for suppliers: {missing}')
    for supplier in suppliers:
        if set(cost[supplier].keys()) != set(customers):
            missing = set(customers) - set(cost[supplier].keys())
            raise ValueError(f'Missing cost data for supplier {supplier} and customers: {missing}')
    M = sum(demand.values())
    m = gurobipy.Model('UFLP')
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    x = m.addVars([(i, j) for i in suppliers for j in customers], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in customers)) + gurobipy.quicksum((fixed_cost[i] * y[i] for i in suppliers))
    m.setObjective(obj, GRB.MINIMIZE)
    for j in customers:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in suppliers)) == demand[j], name='')
    for i in suppliers:
        m.addConstr(gurobipy.quicksum((x[i, j] for j in customers)) <= M * y[i], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()