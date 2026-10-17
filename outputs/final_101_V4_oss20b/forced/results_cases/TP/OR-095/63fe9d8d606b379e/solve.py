import gurobipy
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA

    # Locate tables by table_id
    tables = data['tables']
    table0 = next(t for t in tables if t['table_id'] == 'file_0_view_0')
    table1 = next(t for t in tables if t['table_id'] == 'file_1_view_0')

    # Parse resource limits
    L = {}
    for rec in table0['records']:
        res = rec['values']['Resource'].strip()
        limit = float(rec['values']['MonthlyLimit'].strip())
        L[res] = limit
    for res in ['LaborHours', 'MaterialA', 'MaterialB']:
        if res not in L:
            raise ValueError(f"Missing resource limit for {res}")
    L_LaborHours = L['LaborHours']
    L_MaterialA = L['MaterialA']
    L_MaterialB = L['MaterialB']

    # Parse product data
    P = []
    LaborHours = {}
    MaterialA = {}
    MaterialB = {}
    Profit = {}
    for rec in table1['records']:
        prod = rec['values']['Product'].strip()
        P.append(prod)
        LaborHours[prod] = float(rec['values']['LaborHours'].strip())
        MaterialA[prod] = float(rec['values']['MaterialA'].strip())
        MaterialB[prod] = float(rec['values']['MaterialB'].strip())
        Profit[prod] = float(rec['values']['Profit'].strip())
    for prod in P:
        if prod not in LaborHours or prod not in MaterialA or prod not in MaterialB or prod not in Profit:
            raise ValueError(f"Missing data for product {prod}")

    # Catalyst rate
    CatalystRate = {prod: 5.0 if prod == 'Widget3' else 0.0 for prod in P}

    # Constants
    SaleCap = 1500.0
    SalePrice = 300.0
    DisposalCost = 200.0

    # Create model
    m = gurobipy.Model('WidgetProduction')
    m.setParam("MIPGap", 1e-6)

    # Decision variables
    x = m.addVars(P, lb=0, vtype=GRB.CONTINUOUS, name='')
    s = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='')

    # Objective
    obj = gurobipy.quicksum(Profit[p] * x[p] for p in P) \
          - DisposalCost * gurobipy.quicksum(CatalystRate[p] * x[p] for p in P) \
          + (SalePrice + DisposalCost) * s
    m.setObjective(obj, GRB.MAXIMIZE)

    # Constraints
    m.addConstrs((gurobipy.quicksum(LaborHours[p] * x[p] for p in P) <= L_LaborHours,
                  name='labor'))
    m.addConstrs((gurobipy.quicksum(MaterialA[p] * x[p] for p in P) <= L_MaterialA,
                  name='materialA'))
    m.addConstrs((gurobipy.quicksum(MaterialB[p] * x[p] for p in P) <= L_MaterialB,
                  name='materialB'))
    m.addConstr(s <= gurobipy.quicksum(CatalystRate[p] * x[p] for p in P),
                name='catalyst_sale_limit')
    m.addConstr(s <= SaleCap, name='catalyst_market_cap')

    # Optimize
    m.optimize()

    # Output results if optimal
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')

    return m

m = solve_problem()
