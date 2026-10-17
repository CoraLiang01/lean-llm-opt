import gurobipy
from gurobipy import GRB

# Extract data from CSVQA_DATA
tables = CSVQA_DATA["tables"]

# Helper to find a table by its id
def get_table(table_id):
    for t in tables:
        if t["table_id"] == table_id:
            return t
    raise ValueError(f"Table {table_id} not found")

# Plant data (file_0_view_0)
plant_table = get_table("file_0_view_0")
plant_records = plant_table["records"]

# Customer data (file_1_view_0)
cust_table = get_table("file_1_view_0")
cust_records = cust_table["records"]

# Build index sets
I = []  # plant ids
J = []  # customer ids

# Dictionaries for parameters
fixed_cost = {}
capacity = {}
transport_cost = {}
demand = {}

# Process plant records
for rec in plant_records:
    vals = rec["values"]
    plant_id = vals["plant"]
    if plant_id in I:
        raise ValueError(f"Duplicate plant id {plant_id}")
    I.append(plant_id)
    try:
        fixed_cost[plant_id] = float(vals["fixed_cost"])
    except KeyError:
        raise ValueError(f"Missing fixed_cost for plant {plant_id}")
    try:
        capacity[plant_id] = float(vals["capacity"])
    except KeyError:
        raise ValueError(f"Missing capacity for plant {plant_id}")
    # Transport costs for each customer
    for cust in J:
        # Will be filled later after J is known
        pass

# Process customer records
for rec in cust_records:
    vals = rec["values"]
    cust_id = vals["customer"]
    if cust_id in J:
        raise ValueError(f"Duplicate customer id {cust_id}")
    J.append(cust_id)
    try:
        demand[cust_id] = float(vals["demand"])
    except KeyError:
        raise ValueError(f"Missing demand for customer {cust_id}")

# Now that we have J, fill transport_cost
for rec in plant_records:
    vals = rec["values"]
    plant_id = vals["plant"]
    for cust in J:
        if cust not in vals:
            raise ValueError(f"Missing transport cost for plant {plant_id} to customer {cust}")
        transport_cost[(plant_id, cust)] = float(vals[cust])

# Validate that all plants have transport costs for all customers
for plant_id in I:
    for cust_id in J:
        if (plant_id, cust_id) not in transport_cost:
            raise ValueError(f"Transport cost missing for plant {plant_id} to customer {cust_id}")

# Validate that all customers have demand
for cust_id in J:
    if cust_id not in demand:
        raise ValueError(f"Demand missing for customer {cust_id}")

# Build the model
def solve_problem():
    m = gurobipy.Model("facility_location")
    m.setParam("MIPGap", 1e-6)

    # Decision variables
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    x = m.addVars([(i, j) for i in I for j in J], vtype=GRB.INTEGER, lb=0, name='')

    # Objective: fixed costs + transport costs
    obj = gurobipy.quicksum(fixed_cost[i] * y[i] for i in I) + \
          gurobipy.quicksum(transport_cost[(i, j)] * x[i, j] for i in I for j in J)
    m.setObjective(obj, GRB.MINIMIZE)

    # Capacity constraints
    m.addConstrs(
        (gurobipy.quicksum(x[i, j] for j in J) <= capacity[i] * y[i], name='')
        for i in I
    )

    # Demand constraints
    m.addConstrs(
        (gurobipy.quicksum(x[i, j] for i in I) >= demand[j], name='')
        for j in J
    )

    m.optimize()
    return m

# Create the model and solve
m = solve_problem()