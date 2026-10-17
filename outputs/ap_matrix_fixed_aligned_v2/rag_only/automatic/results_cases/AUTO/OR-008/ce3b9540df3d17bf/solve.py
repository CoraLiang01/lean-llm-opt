LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "Supplier1",\n      "Customer1": "2",\n      "Customer2": "3",\n      "Customer3": "1",\n      "Customer4": "2",\n      "Customer5": "3",\n      "Customer6": "2"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "Supplier2",\n      "Customer1": "1",\n      "Customer2": "2",\n      "Customer3": "3",\n      "Customer4": "2",\n      "Customer5": "3",\n      "Customer6": "2"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "Supplier3",\n      "Customer1": "3",\n      "Customer2": "1",\n      "Customer3": "2",\n      "Customer4": "3",\n      "Customer5": "2",\n      "Customer6": "3"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "Supplier4",\n      "Customer1": "2",\n      "Customer2": "3",\n      "Customer3": "2",\n      "Customer4": "1",\n      "Customer5": "3",\n      "Customer6": "4"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "Supplier5",\n      "Customer1": "3",\n      "Customer2": "2",\n      "Customer3": "3",\n      "Customer4": "3",\n      "Customer5": "2",\n      "Customer6": "3"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv', 'values': {'Unnamed: 0': 'Supplier1', 'Customer1': '2', 'Customer2': '3', 'Customer3': '1', 'Customer4': '2', 'Customer5': '3', 'Customer6': '2'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv', 'values': {'Unnamed: 0': 'Supplier2', 'Customer1': '1', 'Customer2': '2', 'Customer3': '3', 'Customer4': '2', 'Customer5': '3', 'Customer6': '2'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv', 'values': {'Unnamed: 0': 'Supplier3', 'Customer1': '3', 'Customer2': '1', 'Customer3': '2', 'Customer4': '3', 'Customer5': '2', 'Customer6': '3'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv', 'values': {'Unnamed: 0': 'Supplier4', 'Customer1': '2', 'Customer2': '3', 'Customer3': '2', 'Customer4': '1', 'Customer5': '3', 'Customer6': '4'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv', 'values': {'Unnamed: 0': 'Supplier5', 'Customer1': '3', 'Customer2': '2', 'Customer3': '3', 'Customer4': '3', 'Customer5': '2', 'Customer6': '3'}}]
from gurobipy import Model, GRB, quicksum

def build_and_solve_transportation_model():
    cost_records = [r for r in LEGACY_RECORDS if r['source'].endswith('transportation_costs.csv')]
    suppliers = [r['values']['Unnamed: 0'] for r in cost_records]
    customers = [k for k in cost_records[0]['values'] if k != 'Unnamed: 0']
    cost = {s: {} for s in suppliers}
    for r in cost_records:
        s = r['values']['Unnamed: 0']
        for c in customers:
            cost[s][c] = float(r['values'][c])
    demand = {'Customer1': 70, 'Customer2': 80, 'Customer3': 60, 'Customer4': 90, 'Customer5': 85, 'Customer6': 95}
    supply = {'Supplier1': 200, 'Supplier2': 250, 'Supplier3': 230, 'Supplier4': 220, 'Supplier5': 210}
    if set(suppliers) != set(supply.keys()):
        raise ValueError('Supplier identifiers in supply and cost records do not match')
    if set(customers) != set(demand.keys()):
        raise ValueError('Customer identifiers in demand and cost records do not match')
    for s in suppliers:
        if set(cost[s].keys()) != set(customers):
            raise ValueError(f'Cost record for {s} missing some customers')
    m = Model()
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(quicksum((cost[s][c] * x_vars[s, c] for s in suppliers for c in customers)), GRB.MINIMIZE)
    for c in customers:
        m.addConstr(quicksum((x_vars[s, c] for s in suppliers)) == demand[c], name='')
    for s in suppliers:
        m.addConstr(quicksum((x_vars[s, c] for c in customers)) <= supply[s], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for s in suppliers:
            for c in customers:
                v = x_vars[s, c]
                print(f'{v.VarName} {v.X}')
    else:
        print(m.Status)
    return m
m = build_and_solve_transportation_model()