LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "MOUNT AYR",\n      "CLARINDA": "694.6799999999999",\n      "FORT MADISON": "17.48",\n      "SIOUX CITY": "20.07",\n      "TOLEDO": "199.02",\n      "BANCROFT": "1685.53"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "WAUKEE",\n      "CLARINDA": "15.13",\n      "FORT MADISON": "1.5",\n      "SIOUX CITY": "1.43",\n      "TOLEDO": "27.88",\n      "BANCROFT": "90.69"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "WAVERLY",\n      "CLARINDA": "2.34",\n      "FORT MADISON": "349.34",\n      "SIOUX CITY": "246.6",\n      "TOLEDO": "41.3",\n      "BANCROFT": "78.73"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "PELLA",\n      "CLARINDA": "1181.6",\n      "FORT MADISON": "1458.53",\n      "SIOUX CITY": "1646.36",\n      "TOLEDO": "1924.55",\n      "BANCROFT": "38.93"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "DES MOINES",\n      "CLARINDA": "1030.8",\n      "FORT MADISON": "43.48",\n      "SIOUX CITY": "932.4299999999999",\n      "TOLEDO": "55.39",\n      "BANCROFT": "103.84"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv', 'values': {'Unnamed: 0': 'MOUNT AYR', 'CLARINDA': '694.6799999999999', 'FORT MADISON': '17.48', 'SIOUX CITY': '20.07', 'TOLEDO': '199.02', 'BANCROFT': '1685.53'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv', 'values': {'Unnamed: 0': 'WAUKEE', 'CLARINDA': '15.13', 'FORT MADISON': '1.5', 'SIOUX CITY': '1.43', 'TOLEDO': '27.88', 'BANCROFT': '90.69'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv', 'values': {'Unnamed: 0': 'WAVERLY', 'CLARINDA': '2.34', 'FORT MADISON': '349.34', 'SIOUX CITY': '246.6', 'TOLEDO': '41.3', 'BANCROFT': '78.73'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv', 'values': {'Unnamed: 0': 'PELLA', 'CLARINDA': '1181.6', 'FORT MADISON': '1458.53', 'SIOUX CITY': '1646.36', 'TOLEDO': '1924.55', 'BANCROFT': '38.93'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv', 'values': {'Unnamed: 0': 'DES MOINES', 'CLARINDA': '1030.8', 'FORT MADISON': '43.48', 'SIOUX CITY': '932.4299999999999', 'TOLEDO': '55.39', 'BANCROFT': '103.84'}}]
from gurobipy import Model, GRB, quicksum

def build_model():
    supplier_names = []
    store_names = []
    transportation_costs = {}
    for rec in LEGACY_RECORDS:
        if rec['source'].endswith('transportation_costs.csv'):
            supplier = rec['values']['Unnamed: 0']
            supplier_names.append(supplier)
            for store, cost in rec['values'].items():
                if store == 'Unnamed: 0':
                    continue
                if store not in store_names:
                    store_names.append(store)
                transportation_costs[supplier, store] = float(cost)
    supplier_names = list(dict.fromkeys(supplier_names))
    store_names = list(dict.fromkeys(store_names))
    fixed_costs_dict = {'MOUNT AYR': 96.58, 'WAUKEE': 94.06, 'WAVERLY': 94.37, 'PELLA': 82.88, 'DES MOINES': 94.96}
    demand_dict = {'CLARINDA': 2397, 'FORT MADISON': 1889, 'SIOUX CITY': 2518, 'TOLEDO': 3218, 'BANCROFT': 1813}
    if set(supplier_names) != set(fixed_costs_dict.keys()):
        raise ValueError('Supplier names in records do not match fixed cost keys')
    if set(store_names) != set(demand_dict.keys()):
        raise ValueError('Store names in records do not match demand keys')
    for s in supplier_names:
        for t in store_names:
            if (s, t) not in transportation_costs:
                raise ValueError(f'Missing transportation cost for ({s},{t})')
    M = sum((demand_dict[store] for store in store_names))
    m = Model()
    m.Params.MIPGap = 0.0001
    y_vars = m.addVars(supplier_names, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x_vars = m.addVars(supplier_names, store_names, vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(quicksum((fixed_costs_dict[s] * y_vars[s] for s in supplier_names)) + quicksum((transportation_costs[s, t] * x_vars[s, t] for s in supplier_names for t in store_names)), GRB.MINIMIZE)
    for t in store_names:
        m.addConstr(quicksum((x_vars[s, t] for s in supplier_names)) == demand_dict[t], name='demand_' + t)
    for s in supplier_names:
        m.addConstr(quicksum((x_vars[s, t] for t in store_names)) <= M * y_vars[s], name='activate_' + s)
    return m
m = build_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('ObjVal', m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print('Solver status:', m.Status)