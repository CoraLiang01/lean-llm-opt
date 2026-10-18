LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S1",\n      "C1": "0.08",\n      "C2": "52.33",\n      "C3": "73.56999999999999",\n      "C4": "1237.33",\n      "C5": "0.07000000000000001",\n      "C6": "112.16"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S2",\n      "C1": "46.02",\n      "C2": "175.23",\n      "C3": "2026.83",\n      "C4": "299.89",\n      "C5": "966.53",\n      "C6": "1590.42"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S3",\n      "C1": "1031.74",\n      "C2": "78.13",\n      "C3": "99.02",\n      "C4": "277.07",\n      "C5": "884.45",\n      "C6": "1800.86"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S4",\n      "C1": "868.75",\n      "C2": "94.2",\n      "C3": "1776.34",\n      "C4": "285.48",\n      "C5": "868.85",\n      "C6": "86.55"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S5",\n      "C1": "1577",\n      "C2": "760.15",\n      "C3": "2090.19",\n      "C4": "43.2",\n      "C5": "1577.12",\n      "C6": "1095.17"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S6",\n      "C1": "49.14",\n      "C2": "4.33",\n      "C3": "2079.57",\n      "C4": "277.04",\n      "C5": "1032.01",\n      "C6": "1543.49"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '0.08', 'C2': '52.33', 'C3': '73.56999999999999', 'C4': '1237.33', 'C5': '0.07000000000000001', 'C6': '112.16'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '46.02', 'C2': '175.23', 'C3': '2026.83', 'C4': '299.89', 'C5': '966.53', 'C6': '1590.42'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv', 'values': {'Unnamed: 0': 'S3', 'C1': '1031.74', 'C2': '78.13', 'C3': '99.02', 'C4': '277.07', 'C5': '884.45', 'C6': '1800.86'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv', 'values': {'Unnamed: 0': 'S4', 'C1': '868.75', 'C2': '94.2', 'C3': '1776.34', 'C4': '285.48', 'C5': '868.85', 'C6': '86.55'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv', 'values': {'Unnamed: 0': 'S5', 'C1': '1577', 'C2': '760.15', 'C3': '2090.19', 'C4': '43.2', 'C5': '1577.12', 'C6': '1095.17'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv', 'values': {'Unnamed: 0': 'S6', 'C1': '49.14', 'C2': '4.33', 'C3': '2079.57', 'C4': '277.04', 'C5': '1032.01', 'C6': '1543.49'}}]
from gurobipy import Model, GRB, quicksum

def build_and_solve():
    suppliers = []
    stores = []
    transportation_costs = {}
    for rec in LEGACY_RECORDS:
        row = rec['values']
        supplier = row['Unnamed: 0']
        if supplier not in suppliers:
            suppliers.append(supplier)
        for store in row:
            if store != 'Unnamed: 0':
                if store not in stores:
                    stores.append(store)
    suppliers = list(suppliers)
    stores = list(stores)
    fixed_costs = {'S1': 98.88, 'S2': 99.73, 'S3': 94.01, 'S4': 93.77, 'S5': 107.59, 'S6': 112.65}
    demands = {'C1': 216, 'C2': 216, 'C3': 216, 'C4': 144, 'C5': 144, 'C6': 144}
    transportation_costs = {}
    for rec in LEGACY_RECORDS:
        row = rec['values']
        supplier = row['Unnamed: 0']
        for store in stores:
            val = row[store]
            try:
                cost = float(val)
            except Exception:
                raise ValueError(f'Invalid cost for {supplier},{store}: {val}')
            transportation_costs[supplier, store] = cost
    for i in suppliers:
        if i not in fixed_costs:
            raise ValueError(f'Missing fixed cost for supplier {i}')
        for j in stores:
            if (i, j) not in transportation_costs:
                raise ValueError(f'Missing transportation cost for ({i},{j})')
    for j in stores:
        if j not in demands:
            raise ValueError(f'Missing demand for store {j}')
    m = Model()
    y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    x_vars = m.addVars(suppliers, stores, vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(quicksum((fixed_costs[i] * y_vars[i] for i in suppliers)) + quicksum((transportation_costs[i, j] * x_vars[i, j] for i in suppliers for j in stores)), GRB.MINIMIZE)
    for j in stores:
        m.addConstr(quicksum((x_vars[i, j] for i in suppliers)) == demands[j], name='')
    for i in suppliers:
        for j in stores:
            m.addConstr(x_vars[i, j] <= demands[j] * y_vars[i], name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for v in m.getVars():
            print(v.VarName, v.X)
    else:
        print(m.Status)
    return m
m = build_and_solve()