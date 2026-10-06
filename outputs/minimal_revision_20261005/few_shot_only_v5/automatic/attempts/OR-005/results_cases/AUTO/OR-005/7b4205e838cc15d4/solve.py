import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/customer_demand.csv'
    supply_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/supply_capacity.csv'
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/transportation_costs.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            demand_df = pd.read_csv(demand_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Failed to read {demand_path} with tried encodings.')
    for enc in encodings:
        try:
            supply_df = pd.read_csv(supply_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Failed to read {supply_path} with tried encodings.')
    for enc in encodings:
        try:
            cost_df = pd.read_csv(cost_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Failed to read {cost_path} with tried encodings.')
    customers = demand_df['Customers'].astype(str).tolist()
    suppliers = supply_df['Supplier'].astype(str).tolist()
    demand = {}
    for (_, row) in demand_df.iterrows():
        cust = str(row['Customers'])
        val = row['demand']
        if cust in demand:
            demand[cust] += val
        else:
            demand[cust] = val
    supply_capacity = {}
    for (_, row) in supply_df.iterrows():
        sup = str(row['Supplier'])
        val = row['supply_capacity']
        if sup in supply_capacity:
            supply_capacity[sup] += val
        else:
            supply_capacity[sup] = val
    if cost_df.columns[0].casefold() in ['supplier', 'suppliers', 'unnamed: 0']:
        cost_df = cost_df.rename(columns={cost_df.columns[0]: 'Supplier'})
    cost = {}
    for (_, row) in cost_df.iterrows():
        sup = str(row['Supplier'])
        cost[sup] = {}
        for cust in customers:
            if cust not in row:
                raise ValueError(f'Cost data missing for supplier {sup}, customer {cust}')
            cost[sup][cust] = row[cust]
    for cust in customers:
        if cust not in demand:
            raise ValueError(f'Demand data missing for customer {cust}')
    for sup in suppliers:
        if sup not in supply_capacity:
            raise ValueError(f'Supply capacity data missing for supplier {sup}')
        if sup not in cost:
            raise ValueError(f'Cost data missing for supplier {sup}')
        for cust in customers:
            if cust not in cost[sup]:
                raise ValueError(f'Cost data missing for supplier {sup}, customer {cust}')
    m = gp.Model('Amazon_Distribution_TP')
    m.Params.MIPGap = 0.0001
    keys = [(sup, cust) for sup in suppliers for cust in customers]
    x = m.addVars(keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[sup][cust] * x[sup, cust] for sup in suppliers for cust in customers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[sup, cust] for sup in suppliers)) >= demand[cust] for cust in customers), name='')
    m.addConstrs((gp.quicksum((x[sup, cust] for cust in customers)) <= supply_capacity[sup] for sup in suppliers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()