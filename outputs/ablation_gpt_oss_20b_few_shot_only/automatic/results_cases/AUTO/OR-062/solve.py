import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    demand_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv'
    fixed_cost_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv'
    transport_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv'
    demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
    if demand_df.shape[1] < 2:
        raise ValueError('Demand file must have at least two columns')
    store_names = demand_df.iloc[:, 0].tolist()
    try:
        demand_values = demand_df.iloc[:, 1].astype(float).tolist()
    except ValueError as e:
        raise ValueError(f'Non-numeric demand value: {e}')
    demand = dict(zip(store_names, demand_values))
    fixed_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
    if fixed_df.shape[1] < 2:
        raise ValueError('Fixed cost file must have at least two columns')
    supplier_names = fixed_df.iloc[:, 0].tolist()
    try:
        fixed_costs = fixed_df.iloc[:, 1].astype(float).tolist()
    except ValueError as e:
        raise ValueError(f'Non-numeric fixed cost value: {e}')
    fixed = dict(zip(supplier_names, fixed_costs))
    transport_df = pd.read_csv(transport_path, dtype=str, keep_default_na=False)
    if transport_df.shape[1] < 2:
        raise ValueError('Transportation cost file must have at least two columns')
    transport_suppliers = transport_df.iloc[:, 0].tolist()
    transport_stores = transport_df.columns[1:].tolist()
    c = {}
    for (idx, sup) in enumerate(transport_suppliers):
        for col in transport_stores:
            val_str = transport_df.iloc[idx, transport_df.columns.get_loc(col)]
            try:
                val = float(val_str)
            except ValueError as e:
                raise ValueError(f'Non-numeric transportation cost for supplier {sup}, store {col}: {e}')
            c[sup, col] = val
    for sup in supplier_names:
        if sup not in fixed:
            raise ValueError(f'Missing fixed cost for supplier {sup}')
        if sup not in transport_suppliers:
            raise ValueError(f'Supplier {sup} missing in transportation cost data')
    for sup in transport_suppliers:
        if sup not in supplier_names:
            raise ValueError(f'Supplier {sup} present in transportation cost data but missing in fixed cost data')
    for store in store_names:
        if store not in demand:
            raise ValueError(f'Missing demand for store {store}')
        if store not in transport_stores:
            raise ValueError(f'Store {store} missing in transportation cost data')
    for store in transport_stores:
        if store not in store_names:
            raise ValueError(f'Store {store} present in transportation cost data but missing in demand data')
    for sup in supplier_names:
        for store in store_names:
            if (sup, store) not in c:
                raise ValueError(f'Cost missing for supplier {sup} to store {store}')
    M = sum(demand.values())
    m = gurobipy.Model('UFLP')
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(supplier_names, vtype=GRB.BINARY, name='')
    x = m.addVars([(i, j) for i in supplier_names for j in store_names], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((fixed[i] * y[i] for i in supplier_names)) + gurobipy.quicksum((c[i, j] * x[i, j] for i in supplier_names for j in store_names))
    m.setObjective(obj, GRB.MINIMIZE)
    for j in store_names:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in supplier_names)) == demand[j], name='')
    for i in supplier_names:
        m.addConstr(gurobipy.quicksum((x[i, j] for j in store_names)) <= M * y[i], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()