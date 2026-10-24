import gurobipy
from gurobipy import GRB

# Extract tables from CSVQA_DATA
tables = CSVQA_DATA["tables"]

# Demand table
demand_table = next(t for t in tables if t["table_id"] == "file_0_view_0")
demand = {}
for rec in demand_table["records"]:
    cust = rec["values"]["customer"]
    demand[cust] = float(rec["values"]["demand"])

# Fixed cost table
fixed_table = next(t for t in tables if t["table_id"] == "file_1_view_0")
fixed_cost = {}
for rec in fixed_table["records"]:
    sup = rec["values"]["Unnamed: 0"]
    fixed_cost[sup] = float(rec["values"]["fixed_costs"])

# Transportation cost table
cost_table = next(t for t in tables if t["table_id"] == "file_2_view_0")
cost = {}
suppliers = []
for rec in cost_table["records"]:
    sup = rec["values"]["Unnamed: 0"]
    suppliers.append(sup)
    cost[sup] = {}
    for col in cost_table["columns"]:
        if col == "Unnamed: 0":
            continue
        cost[sup][col] = float(rec["values"][col])

# Dealerships (customers) that have cost data
J = set(col for col in cost_table["columns"] if col != "Unnamed: 0")

# Validate data coverage
for sup in suppliers:
    if sup not in fixed_cost:
        raise ValueError(f"Missing fixed cost for supplier {sup}")
for sup in fixed_cost:
    if sup not in cost:
        raise ValueError(f"Missing cost data for supplier {sup}")
for cust in J:
    if cust not in demand:
        raise ValueError(f"Missing demand for customer {cust}")

# Big-M constant
M = sum(demand[cust] for cust in J)

# Create model
m = gurobipy.Model("Supplier_Opening_TP")
m.setParam("MIPGap", 1e-6)

# Decision variables
y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
x = m.addVars((i, j) for i in suppliers for j in J,
               vtype=GRB.CONTINUOUS, lb=0, name='')

# Objective function
obj = gurobipy.quicksum(fixed_cost[i] * y[i] for i in suppliers) + \
      gurobipy.quicksum(cost[i][j] * x[i, j] for i in suppliers for j in J)
m.setObjective(obj, GRB.MINIMIZE)

# Demand satisfaction constraints
m.addConstrs((gurobipy.quicksum(x[i, j] for i in suppliers) == demand[j]
              for j in J), name='demand')

# Supplier activation (big‑M) constraints
m.addConstrs((x[i, j] <= M * y[i] for i in suppliers for j in J), name='bigM')

# Optimize
m.optimize()
