LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S1",\n      "C1": "1506.22",\n      "C2": "70.90000000000001",\n      "C3": "8.44"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S2",\n      "C1": "1732.65",\n      "C2": "1780.72",\n      "C3": "567.4400000000001"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S3",\n      "C1": "115.66",\n      "C2": "100.76",\n      "C3": "64.68000000000001"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '1506.22', 'C2': '70.90000000000001', 'C3': '8.44'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '1732.65', 'C2': '1780.72', 'C3': '567.4400000000001'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S3', 'C1': '115.66', 'C2': '100.76', 'C3': '64.68000000000001'}}]
from gurobipy import Model, GRB, quicksum

def build_and_solve_bandcamp_warehouse_model():
    global LEGACY_RECORDS
    facilities = []
    customers = []
    transportation_cost = {}
    for rec in LEGACY_RECORDS:
        if rec['source'].endswith('transportation_costs.csv'):
            facility = rec['values']['Unnamed: 0']
            if facility not in facilities:
                facilities.append(facility)
            for cust in rec['values']:
                if cust != 'Unnamed: 0':
                    if cust not in customers:
                        customers.append(cust)
                    transportation_cost[facility, cust] = float(rec['values'][cust])
    facilities = sorted(facilities)
    customers = sorted(customers)
    fixed_cost = {}
    demand = {}
    for rec in LEGACY_RECORDS:
        if rec['source'].endswith('fixed_cost.csv'):
            facility = rec['values']['Unnamed: 0']
            fixed_cost[facility] = float(rec['values']['fixed_cost'])
        if rec['source'].endswith('demand.csv'):
            customer = rec['values']['Unnamed: 0']
            demand[customer] = float(rec['values']['demand'])
    if len(fixed_cost) == 0:
        fixed_cost = {'S1': 102.33, 'S2': 94.92, 'S3': 91.83}
    if len(demand) == 0:
        demand = {'C1': 1083, 'C2': 776, 'C3': 16214}
    for f in facilities:
        if f not in fixed_cost:
            raise ValueError(f'Missing fixed cost for facility {f}')
    for c in customers:
        if c not in demand:
            raise ValueError(f'Missing demand for customer {c}')
    for f in facilities:
        for c in customers:
            if (f, c) not in transportation_cost:
                raise ValueError(f'Missing transportation cost for facility {f}, customer {c}')
    m = Model()
    m.Params.MIPGap = 0.0001
    y_vars = m.addVars(facilities, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x_vars = m.addVars(facilities, customers, vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(quicksum((fixed_cost[f] * y_vars[f] for f in facilities)) + quicksum((transportation_cost[f, c] * x_vars[f, c] for f in facilities for c in customers)), GRB.MINIMIZE)
    for c in customers:
        m.addConstr(quicksum((x_vars[f, c] for f in facilities)) == demand[c], name='demand_' + c)
    for f in facilities:
        for c in customers:
            m.addConstr(x_vars[f, c] <= demand[c] * y_vars[f], name='link_%s_%s' % (f, c))
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for v in m.getVars():
            print(v.VarName, v.X)
    else:
        print(m.Status)
    return m
m = build_and_solve_bandcamp_warehouse_model()