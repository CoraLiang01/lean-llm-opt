import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    path_cost = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/expanded_customer_service_costs.csv'
    path_fixed = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/service_centers_fixed_costs.csv'
    df_cost = pd.read_csv(path_cost, dtype=str, keep_default_na=False)
    customers = [c.strip() for c in df_cost.iloc[:, 0].tolist()]
    service_centers = [c.strip() for c in df_cost.columns[1:]]
    cost = {}
    for (_, row) in df_cost.iterrows():
        cust = row.iloc[0].strip()
        for col in df_cost.columns[1:]:
            sc = col.strip()
            val_str = row[col].strip()
            if val_str == '':
                continue
            cost.setdefault(sc, {})[cust] = float(val_str)
    df_fixed = pd.read_csv(path_fixed, dtype=str, keep_default_na=False)
    df_fixed = df_fixed.dropna(subset=[df_fixed.columns[0]])
    df_fixed = df_fixed[~df_fixed.iloc[:, 0].str.contains('Service Center Fixed Opening Cost', na=False)]
    fixed_cost = {}
    for (_, row) in df_fixed.iterrows():
        sc = row.iloc[0].strip()
        val_str = row.iloc[1].strip()
        if val_str == '':
            continue
        fixed_cost[sc] = float(val_str)
    for sc in service_centers:
        if sc not in fixed_cost:
            raise ValueError(f'Missing fixed cost for service center {sc}')
    for sc in service_centers:
        if sc not in cost:
            raise ValueError(f'Missing cost entries for service center {sc}')
        for cust in customers:
            if cust not in cost[sc]:
                raise ValueError(f'Missing cost for service center {sc} and customer {cust}')
    m = gurobipy.Model('UFLP')
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(service_centers, vtype=GRB.BINARY, name='')
    a = m.addVars([(i, j) for i in service_centers for j in customers], vtype=GRB.BINARY, name='')
    obj = gurobipy.quicksum((fixed_cost[i] * y[i] for i in service_centers)) + gurobipy.quicksum((cost[i][j] * a[i, j] for i in service_centers for j in customers))
    m.setObjective(obj, GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((a[i, j] for i in service_centers)) == 1 for j in customers), name='')
    m.addConstrs((a[i, j] <= y[i] for i in service_centers for j in customers), name='')
    m.addConstrs((gurobipy.quicksum((a[i, j] for j in customers)) <= 4 * y[i] for i in service_centers), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')