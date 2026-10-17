import gurobipy
from gurobipy import GRB

def solve_problem():
    # Extract tables from CSVQA_DATA
    tables = CSVQA_DATA["tables"]
    cost_table = next(t for t in tables if t["table_id"] == "file_0_view_0")
    warehouse_table = next(t for t in tables if t["table_id"] == "file_1_view_0")
    demand_table = next(t for t in tables if t["table_id"] == "file_2_view_0")

    # Index sets
    W = [rec["values"]["Warehouse ID"] for rec in warehouse_table["records"]]
    C = [rec["values"]["Customer ID"] for rec in demand_table["records"]]

    # Parameters for warehouses
    f_w = {}
    s_w = {}
    for rec in warehouse_table["records"]:
        w = rec["values"]["Warehouse ID"]
        f_w[w] = float(rec["values"]["Fixed_Cost"])
        s_w[w] = float(rec["values"]["Capacity"])
    if not set(W).issubset(f_w.keys()):
        missing = set(W) - f_w.keys()
        raise ValueError(f"Missing fixed cost for warehouses: {missing}")
    if not set(W).issubset(s_w.keys()):
        missing = set(W) - s_w.keys()
        raise ValueError(f"Missing capacity for warehouses: {missing}")

    # Parameters for customers
    d_c = {}
    for rec in demand_table["records"]:
        c = rec["values"]["Customer ID"]
        d_c[c] = float(rec["values"]["Demand"])
    if not set(C).issubset(d_c.keys()):
        missing = set(C) - d_c.keys()
        raise ValueError(f"Missing demand for customers: {missing}")

    # Transportation cost matrix
    c_wc = {}
    for rec in cost_table["records"]:
        w = rec["values"]["Warehouse ID"]
        c_wc[w] = {}
        for c in C:
            if c not in rec["values"]:
                raise ValueError(f"Missing cost for warehouse {w} to customer {c}")
            c_wc[w][c] = float(rec["values"][c])
    for w in W:
        if w not in c_wc:
            raise ValueError(f"Missing cost row for warehouse {w}")
        for c in C:
            if c not in c_wc[w]:
                raise ValueError(f"Missing cost for warehouse {w} to customer {c}")

    # Build model
    model = gurobipy.Model()
    model.setParam("MIPGap", 1e-6)

    # Decision variables
    y = model.addVars(W, vtype=GRB.BINARY, name='')
    x = model.addVars((w, c) for w in W for c in C, vtype=GRB.CONTINUOUS, name='')

    # Objective
    obj = gurobipy.quicksum(f_w[w] * y[w] for w in W) + \
          gurobipy.quicksum(c_wc[w][c] * x[w, c] for w in W for c in C)
    model.setObjective(obj, GRB.MINIMIZE)

    # Demand constraints
    model.addConstrs(
        (gurobipy.quicksum(x[w, c] for w in W) >= d_c[c] for c in C),
        name='demand'
    )

    # Capacity constraints
    model.addConstrs(
        (gurobipy.quicksum(x[w, c] for c in C) <= s_w[w] * y[w] for w in W),
        name='capacity'
    )

    # Optimize
    model.optimize()

    # Output results if optimal
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')

    return model

m = solve_problem()