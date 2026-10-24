import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    demand_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv'
    fixed_cost_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv'
    trans_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv'
    df_demand = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
    df_fixed = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
    df_trans = pd.read_csv(trans_path, dtype=str, keep_default_na=False)
    df_demand['customer'] = df_demand['customer'].str.strip()
    df_fixed['Unnamed: 0'] = df_fixed['Unnamed: 0'].str.strip()
    df_trans['Unnamed: 0'] = df_trans['Unnamed: 0'].str.strip()
    df_demand['demand'] = df_demand['demand'].astype(float)
    df_fixed['fixed_costs'] = df_fixed['fixed_costs'].astype(float)
    for col in df_trans.columns[1:]:
        df_trans[col] = df_trans[col].astype(float)
    suppliers = df_fixed['Unnamed: 0'].tolist()
    customers = df_demand['customer'].tolist()
    f = {row['Unnamed: 0']: float(row['fixed_costs']) for (_, row) in df_fixed.iterrows()}
    d = {row['customer']: float(row['demand']) for (_, row) in df_demand.iterrows()}
    c = {}
    for (_, row) in df_trans.iterrows():
        supplier = row['Unnamed: 0']
        c[supplier] = {}
        for customer in customers:
            c[supplier][customer] = float(row[customer])
    if set(c.keys()) != set(suppliers):
        raise ValueError('Supplier identifiers mismatch between fixed cost and transportation data')
    for supplier in suppliers:
        if set(c[supplier].keys()) != set(customers):
            raise ValueError(f'Customer identifiers mismatch for supplier {supplier}')
    for customer in customers:
        if customer not in d:
            raise ValueError(f'Demand missing for customer {customer}')
    M = sum(d.values())
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in suppliers for j in customers], lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    obj = gurobipy.quicksum((c[i][j] * x[i, j] for i in suppliers for j in customers)) + gurobipy.quicksum((f[i] * y[i] for i in suppliers))
    m.setObjective(obj, GRB.MINIMIZE)
    for j in customers:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in suppliers)) == d[j], name='')
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