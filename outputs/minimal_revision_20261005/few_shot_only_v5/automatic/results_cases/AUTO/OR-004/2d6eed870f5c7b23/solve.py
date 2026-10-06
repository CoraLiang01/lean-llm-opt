import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/customer_demand.csv'
    supply_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/supply_capacity.csv'
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/transportation_costs.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            customer_demand = pd.read_csv(demand_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Failed to read {demand_path} with tried encodings.')
    for enc in encodings:
        try:
            supply_capacity = pd.read_csv(supply_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Failed to read {supply_path} with tried encodings.')
    for enc in encodings:
        try:
            transportation_costs = pd.read_csv(cost_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Failed to read {cost_path} with tried encodings.')
    customers = customer_demand['customer'].astype(str).tolist()
    suppliers = supply_capacity.iloc[:, 0].astype(str).tolist()
    demand = {}
    for (_, row) in customer_demand.iterrows():
        cust = str(row['customer'])
        if cust in demand:
            demand[cust] += float(row['demand'])
        else:
            demand[cust] = float(row['demand'])
    supply = {}
    for (_, row) in supply_capacity.iterrows():
        sup = str(row.iloc[0])
        if sup in supply:
            supply[sup] += float(row['supply_capacity'])
        else:
            supply[sup] = float(row['supply_capacity'])
    cost = {}
    for (idx, row) in transportation_costs.iterrows():
        sup = str(row.iloc[0])
        if sup not in suppliers:
            continue
        cost[sup] = {}
        for cust in customers:
            if cust not in transportation_costs.columns:
                raise ValueError(f'Customer {cust} not found in transportation_costs columns.')
            val = row[cust]
            if pd.isnull(val):
                raise ValueError(f'Missing cost for supplier {sup}, customer {cust}.')
            cost[sup][cust] = float(val)
    for cust in customers:
        if cust not in demand:
            raise ValueError(f'Demand for customer {cust} missing.')
    for sup in suppliers:
        if sup not in supply:
            raise ValueError(f'Supply for supplier {sup} missing.')
        if sup not in cost:
            raise ValueError(f'Cost row for supplier {sup} missing.')
        for cust in customers:
            if cust not in cost[sup]:
                raise ValueError(f'Cost for supplier {sup}, customer {cust} missing.')
    keys = [(i, j) for i in suppliers for j in customers]
    m = gp.Model('TP4')
    x = m.addVars(keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for (i, j) in keys)), GRB.MINIMIZE)
    for j in customers:
        m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) >= demand[j], name='d_' + j)
    for i in suppliers:
        m.addConstr(gp.quicksum((x[i, j] for j in customers)) <= supply[i], name='s_' + i)
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