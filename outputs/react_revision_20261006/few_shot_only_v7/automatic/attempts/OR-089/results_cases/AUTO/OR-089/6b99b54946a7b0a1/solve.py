import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_paths = [('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/service_centers_fixed_costs.csv', ['Service Center', 'Fixed Opening Cost']), ('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/expanded_customer_service_costs.csv', ['Customer', 'SC1', 'SC2', 'SC3', 'SC4', 'SC5', 'SC6', 'SC7', 'SC8', 'SC9', 'SC10'])]

    def read_csv(path, columns):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
                if set(columns).issubset(df.columns):
                    return df
            except Exception:
                continue
        raise RuntimeError(f'Could not read {path} with required columns {columns}')
    fixed_costs_df = read_csv(csv_paths[0][0], csv_paths[0][1])
    service_costs_df = read_csv(csv_paths[1][0], csv_paths[1][1])
    service_centers = fixed_costs_df['Service Center'].tolist()
    customers = service_costs_df['Customer'].tolist()
    fixed_cost = {}
    for (idx, row) in fixed_costs_df.iterrows():
        sc = row['Service Center']
        try:
            cost = float(row['Fixed Opening Cost'])
        except Exception:
            raise ValueError(f"Invalid fixed opening cost for {sc}: {row['Fixed Opening Cost']}")
        fixed_cost[sc] = cost
    cost = {sc: {} for sc in service_centers}
    for (idx, row) in service_costs_df.iterrows():
        cust = row['Customer']
        for sc in service_centers:
            try:
                cij = float(row[sc])
            except Exception:
                raise ValueError(f'Invalid service cost for {sc}, {cust}: {row[sc]}')
            cost[sc][cust] = cij
    if set(cost.keys()) != set(service_centers):
        raise ValueError('Mismatch in service center keys between cost and service_centers')
    for sc in service_centers:
        if set(cost[sc].keys()) != set(customers):
            raise ValueError(f'Mismatch in customer keys for {sc} between cost and customers')
    m = gp.Model('UFLP13')
    m.Params.MIPGap = 0.0001
    y_vars = m.addVars(service_centers, vtype=GRB.BINARY, name='')
    x_vars = m.addVars(service_centers, customers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((fixed_cost[sc] * y_vars[sc] for sc in service_centers)) + gp.quicksum((cost[sc][cust] * x_vars[sc, cust] for sc in service_centers for cust in customers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[sc, cust] for sc in service_centers)) == 1 for cust in customers), name='')
    m.addConstrs((x_vars[sc, cust] <= y_vars[sc] for sc in service_centers for cust in customers), name='')
    m.addConstrs((gp.quicksum((x_vars[sc, cust] for cust in customers)) <= 4 * y_vars[sc] for sc in service_centers), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')