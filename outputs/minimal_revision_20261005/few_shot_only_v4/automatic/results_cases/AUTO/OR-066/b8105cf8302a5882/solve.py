import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_with_encodings(path, **kwargs):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with supported encodings.')
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv'
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv'
    trans_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv'
    demand_df = read_csv_with_encodings(demand_path)
    fixed_cost_df = read_csv_with_encodings(fixed_cost_path)
    trans_cost_df = read_csv_with_encodings(trans_cost_path)
    customers = demand_df['customer'].astype(str).tolist()
    suppliers = fixed_cost_df['Unnamed: 0'].astype(str).tolist()
    demand = {}
    for (_, row) in demand_df.iterrows():
        cust = str(row['customer'])
        if cust in demand:
            demand[cust] += float(row['demand'])
        else:
            demand[cust] = float(row['demand'])
    fixed_cost = {}
    for (_, row) in fixed_cost_df.iterrows():
        sup = str(row['Unnamed: 0'])
        if sup in fixed_cost:
            fixed_cost[sup] += float(row['fixed_costs']) if sup in fixed_cost else float(row['fixed_costs'])
        else:
            fixed_cost[sup] = float(row['fixed_costs'])
    cost = {sup: {} for sup in suppliers}
    for (_, row) in trans_cost_df.iterrows():
        sup = str(row['Unnamed: 0'])
        if sup not in suppliers:
            continue
        for cust in customers:
            if cust not in row:
                raise ValueError(f'Missing transportation cost for supplier {sup}, customer {cust}')
            cost[sup][cust] = float(row[cust])
    for cust in customers:
        if cust not in demand:
            raise ValueError(f'Demand missing for customer {cust}')
    for sup in suppliers:
        if sup not in fixed_cost:
            raise ValueError(f'Fixed cost missing for supplier {sup}')
        for cust in customers:
            if cust not in cost[sup]:
                raise ValueError(f'Transportation cost missing for supplier {sup}, customer {cust}')
    M = sum((demand[cust] for cust in customers))
    m = gp.Model('UFLP')
    m.Params.MIPGap = 0.0001
    x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[sup][cust] * x[sup, cust] for sup in suppliers for cust in customers)) + gp.quicksum((fixed_cost[sup] * y[sup] for sup in suppliers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[sup, cust] for sup in suppliers)) == demand[cust] for cust in customers), name='')
    m.addConstrs((gp.quicksum((x[sup, cust] for cust in customers)) <= M * y[sup] for sup in suppliers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()