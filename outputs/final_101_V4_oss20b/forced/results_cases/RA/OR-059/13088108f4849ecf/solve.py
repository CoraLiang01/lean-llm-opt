import gurobipy

def solve_problem():
    # Extract tables from CSVQA_DATA
    tables = CSVQA_DATA["tables"]

    # Suppliers table
    sup_table = next(t for t in tables if t["table_id"] == "file_1_view_0")
    suppliers = {}
    for rec in sup_table["records"]:
        vals = rec["values"]
        sid = vals["Unnamed: 0"]
        suppliers[sid] = float(vals["fixed_costs"])

    # Customers table
    cust_table = next(t for t in tables if t["table_id"] == "file_0_view_0")
    customers = {}
    for rec in cust_table["records"]:
        vals = rec["values"]
        cid = vals["customer"]
        customers[cid] = float(vals["demand"])

    # Transportation costs matrix
    cost_table = next(t for t in tables if t["table_id"] == "file_2_view_0")
    costs = {}
    for rec in cost_table["records"]:
        vals = rec["values"]
        sid = vals["Unnamed: 0"]
        for cid in customers.keys():
            if cid not in vals:
                raise ValueError(f"Missing cost for supplier {sid} to customer {cid}")
            costs[(sid, cid)] = float(vals[cid])

    # Validate that all required keys are present
    for sid in suppliers:
        if sid not in costs:
            raise ValueError(f"Supplier {sid} has no cost entries")
    for cid in customers:
        for sid in suppliers:
            if (sid, cid) not in costs:
                raise ValueError(f"Cost missing for supplier {sid} to customer {cid}")

    # Index sets
    I = list(suppliers.keys())
    J = list(customers.keys())

    # Large constant M
    M = sum(customers[j] for j in J)

    # Create model
    model = gurobipy.Model()
    model.setParam("MIPGap", 1e-6)

    # Decision variables
    y = model.addVars(I, vtype=gurobipy.GRB.BINARY, name='')
    x = model.addVars((i, j) for i in I for j in J,
                      vtype=gurobipy.GRB.CONTINUOUS, lb=0, name='')

    # Objective: minimize fixed costs + transportation costs
    obj = gurobipy.quicksum(suppliers[i] * y[i] for i in I) + \
          gurobipy.quicksum(costs[(i, j)] * x[i, j] for i in I for j in J)
    model.setObjective(obj, gurobipy.GRB.MINIMIZE)

    # Demand constraints
    model.addConstrs(
        (gurobipy.quicksum(x[i, j] for i in I) >= customers[j] for j in J),
        name='demand'
    )

    # Linking constraints
    model.addConstrs(
        (x[i, j] <= M * y[i] for i in I for j in J),
        name='link'
    )

    # Optimize
    model.optimize()

    # Output results if optimal
    if model.Status == gurobipy.GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')

    return model

# Create the global model instance
m = solve_problem()