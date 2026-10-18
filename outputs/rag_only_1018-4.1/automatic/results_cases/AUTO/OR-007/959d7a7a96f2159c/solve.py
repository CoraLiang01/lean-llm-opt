LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S1",\n      "D1": "269.3910588020795",\n      "D2": "1.4537335390933939",\n      "D3": "99.60345345756605",\n      "D4": "26.64078166309837",\n      "D5": "9.537688956880922"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S2",\n      "D1": "9.291846876785183",\n      "D2": "10.874778437070223",\n      "D3": "144.52609291614627",\n      "D4": "11.420133077898234",\n      "D5": "153.1756819927813"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S3",\n      "D1": "9.674584301671008",\n      "D2": "2.6191650959687944",\n      "D3": "100.8242249168735",\n      "D4": "3.2121910887916876",\n      "D5": "133.8493396124168"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S4",\n      "D1": "270.57498480010247",\n      "D2": "32.50253586",\n      "D3": "4.6842098096469815",\n      "D4": "1.5682269686546804",\n      "D5": "9.58927599"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv",\n    "values": {\n      "Unnamed: 0": "S5",\n      "D1": "226.0331910675782",\n      "D2": "8.669161980826471",\n      "D3": "65.47681316968448",\n      "D4": "9.068765258459958",\n      "D5": "202.65015316425533"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'D1': '269.3910588020795', 'D2': '1.4537335390933939', 'D3': '99.60345345756605', 'D4': '26.64078166309837', 'D5': '9.537688956880922'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'D1': '9.291846876785183', 'D2': '10.874778437070223', 'D3': '144.52609291614627', 'D4': '11.420133077898234', 'D5': '153.1756819927813'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S3', 'D1': '9.674584301671008', 'D2': '2.6191650959687944', 'D3': '100.8242249168735', 'D4': '3.2121910887916876', 'D5': '133.8493396124168'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S4', 'D1': '270.57498480010247', 'D2': '32.50253586', 'D3': '4.6842098096469815', 'D4': '1.5682269686546804', 'D5': '9.58927599'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S5', 'D1': '226.0331910675782', 'D2': '8.669161980826471', 'D3': '65.47681316968448', 'D4': '9.068765258459958', 'D5': '202.65015316425533'}}]
from gurobipy import Model, GRB, quicksum

def build_and_solve():
    cost_records = [r for r in LEGACY_RECORDS if r['source'].endswith('transportation_costs.csv')]
    warehouses = [r['values']['Unnamed: 0'] for r in cost_records]
    stores = [k for k in cost_records[0]['values'] if k != 'Unnamed: 0']
    cost = {w: {} for w in warehouses}
    for r in cost_records:
        w = r['values']['Unnamed: 0']
        for d in stores:
            cost[w][d] = float(r['values'][d])
    demand = {'D1': 428, 'D2': 217, 'D3': 214, 'D4': 380, 'D5': 254}
    supply = {'S1': 428, 'S2': 217, 'S3': 214, 'S4': 380, 'S5': 254}
    if set(warehouses) != set(supply.keys()):
        raise ValueError('Warehouse identifiers in cost matrix and supply do not match')
    if set(stores) != set(demand.keys()):
        raise ValueError('Store identifiers in cost matrix and demand do not match')
    for w in warehouses:
        for d in stores:
            if d not in cost[w]:
                raise ValueError(f'Missing cost entry for {w},{d}')
    m = Model()
    x_vars = m.addVars(warehouses, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(quicksum((cost[w][d] * x_vars[w, d] for w in warehouses for d in stores)), GRB.MINIMIZE)
    for d in stores:
        m.addConstr(quicksum((x_vars[w, d] for w in warehouses)) == demand[d], name='')
    for w in warehouses:
        m.addConstr(quicksum((x_vars[w, d] for d in stores)) <= supply[w], name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for w in warehouses:
            for d in stores:
                v = x_vars[w, d]
                print(f'{v.VarName} {v.X}')
    else:
        print(m.Status)
    return m
m = build_and_solve()