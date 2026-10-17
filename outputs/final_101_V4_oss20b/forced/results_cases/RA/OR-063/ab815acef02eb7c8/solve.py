import gurobipy
from gurobipy import GRB

def solve_problem():
    # Extract tables from CSVQA_DATA
    tables = CSVQA_DATA["tables"]
    demand_table = next(t for t in tables if t["table_id"] == "file_0_view_0")
    fixed_cost_table = next(t for t in tables if t["table_id"] == "file_1_view_0")
    transport_table = next(t for t in tables if t["table_id"] == "file_2_view_0")

    # Parse demand
    demand = {}
    for rec in demand_table["records"]:
        customer = rec["values"]["customer"]
        demand_val = float(rec["values"]["demand"])
        demand[customer] = demand_val

    # Parse fixed costs
    fixed_cost = {}
    for rec in fixed_cost_table["records"]:
        warehouse = rec["values"]["Unnamed: 0"]
        cost_val = float(rec["values"]["fixed_costs"])
        fixed_cost[warehouse] = cost_val

    # Parse transportation costs
    transportation = {}
    for rec in transport_table["records"]:
        warehouse = rec["values"]["Unnamed: 0"]
        for customer in transport_table["columns"]:
            if customer == "Unnamed: 0":
                continue
            cost_val = float(rec["values"][customer])
            transportation[(warehouse, customer)] = cost_val

    # Sets of warehouses and customers
    I = list(fixed_cost.keys())
    J = list(demand.keys())

    # Validate data completeness
    for i in I:
        if i not in fixed_cost:
            raise ValueError(f"Missing fixed cost for warehouse {i}")
    for j in J:
        if j not in demand:
            raise ValueError(f"Missing demand for customer {j}")
    for i in I:
        for j in J:
            if (i, j) not in transportation:
                raise ValueError(f"Missing transportation cost for warehouse {i} and customer {j}")

    # Create model
    m = gurobipy.Model()
    m.setParam("MIPGap", 1e-6)

    # Decision variables
    y = m.addVars(I, vtype=GRB.BINARY, name='y')
    x = m.addVars((i, j) for i in I for j in J, vtype=GRB.CONTINUOUS, lb=0, name='x')

    # Objective: fixed costs + transportation costs
    obj = gurobipy.quicksum(fixed_cost[i] * y[i] for i in I) + \
          gurobipy.quicksum(transportation[(i, j)] * x[i, j] for i in I for j in J)
    m.setObjective(obj, GRB.MINIMIZE)

    # Demand satisfaction constraints
    for j in J:
        m.addConstr(gurobipy.quicksum(x[i, j] for i in I) == demand[j], name=f"cust_{j}")

    # Shipping only from active warehouses
    for i in I:
        for j in J:
            m.addConstr(x[i, j] <= demand[j] * y[i], name=f"ship_{i}_{j}")

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

# Instantiate the model at module scope
m = solve_problem()
