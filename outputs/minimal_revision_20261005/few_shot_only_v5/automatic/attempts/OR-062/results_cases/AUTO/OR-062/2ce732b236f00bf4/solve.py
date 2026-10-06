import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            demand_df = pd.read_csv(demand_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {demand_path} with tried encodings.')
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv'
    for enc in encodings:
        try:
            fixed_cost_df = pd.read_csv(fixed_cost_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {fixed_cost_path} with tried encodings.')
    trans_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv'
    for enc in encodings:
        try:
            trans_cost_df = pd.read_csv(trans_cost_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {trans_cost_path} with tried encodings.')
    suppliers = fixed_cost_df['Unnamed: 0'].astype(str).tolist()
    customers = demand_df['Customer'].astype(str).tolist()
    trans_cost_customers = [col for col in trans_cost_df.columns if col != 'Unnamed: 0']
    missing_customers = set(customers) - set(trans_cost_customers)
    if missing_customers:
        raise ValueError(f'Customers {missing_customers} in demand.csv not found in transportation_costs.csv columns.')
    trans_cost_suppliers = trans_cost_df['Unnamed: 0'].astype(str).tolist()
    missing_suppliers = set(suppliers) - set(trans_cost_suppliers)
    if missing_suppliers:
        raise ValueError(f'Suppliers {missing_suppliers} in fixed_cost.csv not found in transportation_costs.csv rows.')
    demand = {}
    for (_, row) in demand_df.iterrows():
        cust = str(row['Customer'])
        val = row['demand']
        if cust in demand:
            demand[cust] += val
        else:
            demand[cust] = val
    fixed_cost = {}
    for (_, row) in fixed_cost_df.iterrows():
        sup = str(row['Unnamed: 0'])
        val = row['fixed_costs']
        if sup in fixed_cost:
            fixed_cost[sup] += val
        else:
            fixed_cost[sup] = val
    cost = {sup: {} for sup in suppliers}
    for (_, row) in trans_cost_df.iterrows():
        sup = str(row['Unnamed: 0'])
        if sup not in suppliers:
            continue
        for cust in customers:
            val = row[cust]
            cost[sup][cust] = val
    for i in suppliers:
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Missing transportation cost for supplier {i}, customer {j}.')
    M = sum((demand[j] for j in customers))
    m = gp.Model('Iowa_Liquor_UFLP')
    m.Params.MIPGap = 0.0001
    x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in customers)) + gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) == demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= M * y[i] for i in suppliers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()