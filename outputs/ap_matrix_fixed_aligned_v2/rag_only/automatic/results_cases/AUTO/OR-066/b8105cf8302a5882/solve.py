LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S1",\n      "C1": "2358.39",\n      "C2": "1492.08"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S2",\n      "C1": "0.07000000000000001",\n      "C2": "52.32"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '2358.39', 'C2': '1492.08'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '0.07000000000000001', 'C2': '52.32'}}]
from gurobipy import Model, GRB, quicksum
supplier_ids = []
customer_ids = []
transportation_costs = {}
for rec in LEGACY_RECORDS:
    if rec['source'].endswith('transportation_costs.csv'):
        supplier = rec['values']['Unnamed: 0']
        supplier_ids.append(supplier)
        for k, v in rec['values'].items():
            if k != 'Unnamed: 0':
                if k not in customer_ids:
                    customer_ids.append(k)
                transportation_costs[supplier, k] = float(v)
supplier_ids = list(dict.fromkeys(supplier_ids))
customer_ids = list(dict.fromkeys(customer_ids))
fixed_costs = {}
demands = {}
for rec in LEGACY_RECORDS:
    if rec['source'].endswith('fixed_cost.csv'):
        supplier = rec['values']['Supplier']
        fixed_costs[supplier] = float(rec['values']['FixedCost'])
    if rec['source'].endswith('demand.csv'):
        customer = rec['values']['Customer']
        demands[customer] = float(rec['values']['Demand'])
if not fixed_costs:
    fixed_costs = {'S1': 105.97, 'S2': 85.31}
if not demands:
    demands = {'C1': 144, 'C2': 216}
for s in supplier_ids:
    if s not in fixed_costs:
        raise ValueError(f'Missing fixed cost for supplier {s}')
for c in customer_ids:
    if c not in demands:
        raise ValueError(f'Missing demand for customer {c}')
for s in supplier_ids:
    for c in customer_ids:
        if (s, c) not in transportation_costs:
            raise ValueError(f'Missing transportation cost for supplier {s}, customer {c}')

def build_model():
    m = Model()
    m.Params.MIPGap = 0.0001
    y_vars = m.addVars(supplier_ids, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x_vars = m.addVars(supplier_ids, customer_ids, vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(quicksum((fixed_costs[s] * y_vars[s] for s in supplier_ids)) + quicksum((transportation_costs[s, c] * x_vars[s, c] for s in supplier_ids for c in customer_ids)), GRB.MINIMIZE)
    for c in customer_ids:
        m.addConstr(quicksum((x_vars[s, c] for s in supplier_ids)) == demands[c], name='demand_' + c)
    for s in supplier_ids:
        for c in customer_ids:
            m.addConstr(x_vars[s, c] <= demands[c] * y_vars[s], name='supply_%s_%s' % (s, c))
    return m
m = build_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')