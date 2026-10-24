import gurobipy
from gurobipy import GRB

def solve_problem():
    # Extract tables by ID
    tables = {t["table_id"]: t for t in CSVQA_DATA["tables"]}

    # Demand table (branches)
    demand_table = tables["file_0_view_0"]
    J = [rec["values"]["customer"] for rec in demand_table["records"]]
    d = {}
    for rec in demand_table["records"]:
        j = rec["values"]["customer"]
        try:
            d[j] = float(rec["values"]["demand"])
        except ValueError:
            raise ValueError(f"Invalid demand value for branch {j}")

    # Fixed cost table (suppliers)
    fixed_table = tables["file_1_view_0"]
    I = [rec["values"]["Unnamed: 0"] for rec in fixed_table["records"]]
    f = {}
    for rec in fixed_table["records"]:
        i = rec["values"]["Unnamed: 0"]
        try:
            f[i] = float(rec["values"]["fixed_costs"])
        except ValueError:
            raise ValueError(f"Invalid fixed cost for supplier {i}")

    # Transportation cost matrix
    trans_table = tables["file_2_view_0"]
    c = {}
    for rec in trans_table["records"]:
        i = rec["values"]["Unnamed: 0"]
        for j in J:
            if j not in rec["values"]:
                raise ValueError(f"Missing transportation cost for supplier {i} to branch {j}")
            try:
                c[(i, j)] = float(rec["values"][j])
            except ValueError:
                raise ValueError(f"Invalid transportation cost for supplier {i} to branch {j}")

    # Validate coverage
    for i in I:
        if i not in f:
            raise ValueError(f"Missing fixed cost for supplier {i}")
    for j in J:
        if j not in d:
            raise ValueError(f"Missing demand for branch {j}")
    for i in I:
        for j in J:
            if (i, j) not in c:
                raise ValueError(f"Missing transportation cost for supplier {i} to branch {j}")

    # Create model
    model = gurobipy.Model()
    model.setParam("MIPGap", 1e-6)

    # Decision variables
    y = model.addVars(I, vtype=GRB.BINARY, name='')
    x = model.addVars((i, j) for i in I for j in J, vtype=GRB.CONTINUOUS, lb=0, name='')

    # Objective
    obj = gurobipy.quicksum(f[i] * y[i] for i in I) + \
          gurobipy.quicksum(c[(i, j)] * x[i, j] for i in I for j in J)
    model.setObjective(obj, GRB.MINIMIZE)

    # Demand constraints
    model.addConstrs((gurobipy.quicksum(x[i, j] for i in I) == d[j] for j in J), name='')

    # Link constraints
    model.addConstrs((x[i, j] <= d[j] * y[i] for i in I for j in J), name='')

    # Optimize
    model.optimize()

    return model

m = solve_problem()
