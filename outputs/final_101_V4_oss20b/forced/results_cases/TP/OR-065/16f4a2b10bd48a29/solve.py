import gurobipy
from gurobipy import GRB

# Extract tables from CSVQA_DATA
tables = CSVQA_DATA["tables"]

# Helper to find a table by its id
def get_table(table_id):
    for t in tables:
        if t["table_id"] == table_id:
            return t
    raise ValueError(f"Table {table_id} not found")

# Demand data (file_0_view_0)
demand_table = get_table("file_0_view_0")
demand = {}
for rec in demand_table["records"]:
    cust = rec["values"]["customer"]
    val = float(rec["values"]["demand"])
    demand[cust] = val

# Fixed cost data (file_1_view_0)
fixed_table = get_table("file_1_view_0")
fixed_cost = {}
for rec in fixed_table["records"]:
    wh = rec["values"]["Unnamed: 0"]
    val = float(rec["values"]["fixed_costs"])
    fixed_cost[wh] = val

# Transportation cost matrix (file_2_view_0)
cost_table = get_table("file_2_view_0")
costs = {}
for rec in cost_table["records"]:
    wh = rec["values"]["Unnamed: 0"]
    costs[wh] = {}
    for col, val in rec["values"].items():
        if col == "Unnamed: 0":
            continue
        costs[wh][col] = float(val)

# Validate indices
I = set(fixed_cost.keys())
J = set(demand.keys())

# Ensure all warehouses have cost entries for all customers
for i in I:
    if i not in costs:
        raise ValueError(f"Missing cost row for warehouse {i}")
    for j in J:
        if j not in costs[i]:
            raise ValueError(f"Missing cost for warehouse {i} to customer {j}")

# Ensure all customers have cost entries from all warehouses
for j in J:
    for i in I:
        if j not in costs[i]:
            raise ValueError(f"Missing cost for warehouse {i} to customer {j}")

# Compute M
M = sum(demand.values())

# Build model
def solve_problem():
    m = gurobipy.Model()
    m.setParam("MIPGap", 1e-6)

    # Decision variables
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    x = m.addVars((i, j) for i in I for j in J, lb=0, vtype=GRB.CONTINUOUS, name='')

    # Objective: fixed costs + transportation costs
    obj = gurobipy.quicksum(fixed_cost[i] * y[i] for i in I)
    obj += gurobipy.quicksum(costs[i][j] * x[i, j] for i in I for j in J)
    m.setObjective(obj, GRB.MINIMIZE)

    # Demand constraints
    m.addConstrs(
        (gurobipy.quicksum(x[i, j] for i in I) >= demand[j] for j in J),
        name='demand'
    )

    # Linking constraints
    m.addConstrs(
        (x[i, j] <= M * y[i] for i in I for j in J),
        name='link'
    )

    # Optimize
    m.optimize()

    # Output results
    if m.Status == GRB.OPTIMAL:
        print(f"ObjVal: {m.ObjVal}")
        for var in m.getVars():
            print(f"{var.VarName}: {var.X}")
    else:
        print(f"Solver status: {m.Status}")

    return m

# Create the global model instance
m = solve_problem()