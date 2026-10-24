import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    cost_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/cost.csv'
    warehouse_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/warehouse.csv'
    demand_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/demand.csv'
    df_cost = pd.read_csv(cost_path, dtype=str, keep_default_na=False)
    warehouse_ids = [w.strip() for w in df_cost.iloc[:, 0]]
    customer_ids = [c.strip() for c in df_cost.columns[1:]]
    cost = {}
    for (idx, w) in enumerate(warehouse_ids):
        cost[w] = {}
        for c in customer_ids:
            val = df_cost.iloc[idx, df_cost.columns.get_loc(c)]
            try:
                cost[w][c] = float(val)
            except ValueError:
                raise ValueError(f'Invalid cost value for warehouse {w}, customer {c}: {val}')
    df_warehouse = pd.read_csv(warehouse_path, dtype=str, keep_default_na=False)
    f_w = {}
    cap_w = {}
    for (_, row) in df_warehouse.iterrows():
        w = row['Warehouse ID'].strip()
        try:
            f_w[w] = float(row['Fixed_Cost'])
        except ValueError:
            raise ValueError(f"Invalid fixed cost for warehouse {w}: {row['Fixed_Cost']}")
        try:
            cap_w[w] = float(row['Capacity'])
        except ValueError:
            raise ValueError(f"Invalid capacity for warehouse {w}: {row['Capacity']}")
    df_demand = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
    d_c = {}
    for (_, row) in df_demand.iterrows():
        c = row['Customer ID'].strip()
        try:
            d_c[c] = float(row['Demand'])
        except ValueError:
            raise ValueError(f"Invalid demand for customer {c}: {row['Demand']}")
    W = warehouse_ids
    C = customer_ids
    for w in W:
        if w not in f_w or w not in cap_w:
            raise ValueError(f'Missing parameters for warehouse {w}')
    for c in C:
        if c not in d_c:
            raise ValueError(f'Missing demand for customer {c}')
    for w in W:
        for c in C:
            if c not in cost[w]:
                raise ValueError(f'Missing cost for warehouse {w}, customer {c}')
    M_w = sum(d_c.values())
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(W, vtype=GRB.BINARY, name='')
    x = m.addVars([(w, c) for w in W for c in C], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((f_w[w] * y[w] for w in W)) + gurobipy.quicksum((cost[w][c] * x[w, c] for w in W for c in C))
    m.setObjective(obj, GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[w, c] for w in W)) == d_c[c] for c in C), name='')
    m.addConstrs((gurobipy.quicksum((x[w, c] for c in C)) <= cap_w[w] for w in W), name='')
    m.addConstrs((x[w, c] <= M_w * y[w] for w in W for c in C), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()