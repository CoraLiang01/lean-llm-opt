LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S1",\n      "C1": "543.756480860856",\n      "C2": "23.685276141764653",\n      "C3": "23.676386730773032",\n      "C4": "447.75143678673766"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S2",\n      "C1": "883.9151090405642",\n      "C2": "0.04977684765576961",\n      "C3": "0.0350986687216299",\n      "C4": "44.45588531711622"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S3",\n      "C1": "537.3456896658107",\n      "C2": "23.769274659075112",\n      "C3": "498.95659249465467",\n      "C4": "440.60737890439776"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S4",\n      "C1": "1791.493192397229",\n      "C2": "68.21633865655126",\n      "C3": "1432.4837339656747",\n      "C4": "1527.7635425462734"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '543.756480860856', 'C2': '23.685276141764653', 'C3': '23.676386730773032', 'C4': '447.75143678673766'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '883.9151090405642', 'C2': '0.04977684765576961', 'C3': '0.0350986687216299', 'C4': '44.45588531711622'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S3', 'C1': '537.3456896658107', 'C2': '23.769274659075112', 'C3': '498.95659249465467', 'C4': '440.60737890439776'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv', 'values': {'Unnamed: 0': 'S4', 'C1': '1791.493192397229', 'C2': '68.21633865655126', 'C3': '1432.4837339656747', 'C4': '1527.7635425462734'}}]
from gurobipy import Model, GRB, quicksum

def build_and_solve():
    plants = []
    customers = []
    cost = {}
    for rec in LEGACY_RECORDS:
        if rec['source'].endswith('transportation_costs.csv'):
            plant = rec['values']['Unnamed: 0']
            if plant not in plants:
                plants.append(plant)
            for cust in rec['values']:
                if cust.startswith('C'):
                    if cust not in customers:
                        customers.append(cust)
                    cost[plant, cust] = float(rec['values'][cust])
    plants.sort()
    customers.sort()
    demand = {'C1': 94, 'C2': 39, 'C3': 65, 'C4': 435}
    supply_capacity = {'S1': 2531, 'S2': 20, 'S3': 210, 'S4': 241}
    for i in plants:
        for j in customers:
            if (i, j) not in cost:
                raise ValueError(f'Missing transportation cost for ({i},{j})')
    for j in customers:
        if j not in demand:
            raise ValueError(f'Missing demand for customer {j}')
    for i in plants:
        if i not in supply_capacity:
            raise ValueError(f'Missing supply capacity for plant {i}')
    m = Model()
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(plants, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(quicksum((cost[i, j] * x_vars[i, j] for i in plants for j in customers)), GRB.MINIMIZE)
    for j in customers:
        m.addConstr(quicksum((x_vars[i, j] for i in plants)) == demand[j], name='')
    for i in plants:
        m.addConstr(quicksum((x_vars[i, j] for j in customers)) <= supply_capacity[i], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for i in plants:
            for j in customers:
                v = x_vars[i, j]
                print(f'{v.VarName} {v.X}')
    else:
        print(m.Status)
    return m
m = build_and_solve()