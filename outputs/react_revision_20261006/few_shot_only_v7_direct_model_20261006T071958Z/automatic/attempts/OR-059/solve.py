import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/demand.csv'
    demand_df = None
    for encoding in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
        try:
            demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False, encoding=encoding)
            break
        except Exception:
            continue
    if demand_df is None:
        raise RuntimeError(f'Failed to read {demand_path} with supported encodings.')
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/fixed_cost.csv'
    fixed_cost_df = None
    for encoding in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
        try:
            fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False, encoding=encoding)
            break
        except Exception:
            continue
    if fixed_cost_df is None:
        raise RuntimeError(f'Failed to read {fixed_cost_path} with supported encodings.')
    trans_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/transportation_costs.csv'
    trans_cost_df = None
    for encoding in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
        try:
            trans_cost_df = pd.read_csv(trans_cost_path, dtype=str, keep_default_na=False, encoding=encoding)
            break
        except Exception:
            continue
    if trans_cost_df is None:
        raise RuntimeError(f'Failed to read {trans_cost_path} with supported encodings.')
    suppliers = fixed_cost_df['Unnamed: 0'].tolist()
    dealerships = demand_df['customer'].tolist()
    demand = {}
    for (_, row) in demand_df.iterrows():
        cust = row['customer']
        try:
            demand[cust] = float(row['demand'])
        except Exception:
            raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
    fixed_cost = {}
    for (_, row) in fixed_cost_df.iterrows():
        sup = row['Unnamed: 0']
        try:
            fixed_cost[sup] = float(row['fixed_costs'])
        except Exception:
            raise ValueError(f"Invalid fixed cost for supplier {sup}: {row['fixed_costs']}")
    cost = {}
    for (_, row) in trans_cost_df.iterrows():
        sup = row['Unnamed: 0']
        cost[sup] = {}
        for j in dealerships:
            if j not in row:
                raise ValueError(f'Dealership {j} not found as column in transportation_costs.csv for supplier {sup}')
            try:
                cost[sup][j] = float(row[j])
            except Exception:
                raise ValueError(f'Invalid transportation cost for supplier {sup}, dealership {j}: {row[j]}')
    for i in suppliers:
        if i not in fixed_cost:
            raise ValueError(f'Supplier {i} missing fixed cost.')
        if i not in cost:
            raise ValueError(f'Supplier {i} missing transportation cost row.')
        for j in dealerships:
            if j not in cost[i]:
                raise ValueError(f'Supplier {i}, dealership {j} missing transportation cost.')
    for j in dealerships:
        if j not in demand:
            raise ValueError(f'Dealership {j} missing demand.')
    M = sum((demand[j] for j in dealerships))
    m = gp.Model('Colorado_Motor_Vehicle_Sales_UFLP')
    m.Params.MIPGap = 0.0001
    x_keys = [(i, j) for i in suppliers for j in dealerships]
    x_vars = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in suppliers for j in dealerships)) + gp.quicksum((fixed_cost[i] * y_vars[i] for i in suppliers)), GRB.MINIMIZE)
    for j in dealerships:
        m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
    for i in suppliers:
        m.addConstr(gp.quicksum((x_vars[i, j] for j in dealerships)) <= M * y_vars[i], name=f'activation_{i}')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')