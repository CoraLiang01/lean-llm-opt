LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv",\n    "values": {\n      "customer": "C1",\n      "demand": "143"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv",\n    "values": {\n      "customer": "C2",\n      "demand": "6"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv",\n    "values": {\n      "customer": "C3",\n      "demand": "10"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv",\n    "values": {\n      "customer": "C4",\n      "demand": "25"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv",\n    "values": {\n      "customer": "C5",\n      "demand": "3"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv', 'values': {'customer': 'C1', 'demand': '143'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv', 'values': {'customer': 'C2', 'demand': '6'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv', 'values': {'customer': 'C3', 'demand': '10'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv', 'values': {'customer': 'C4', 'demand': '25'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv', 'values': {'customer': 'C5', 'demand': '3'}}]
from gurobipy import Model, GRB, quicksum

def build_and_solve():
    suppliers = ['S1', 'S2', 'S3', 'S4', 'S5']
    branches = []
    demand = {}
    for rec in LEGACY_RECORDS:
        if rec['source'].endswith('demand.csv'):
            cust = rec['values']['customer']
            branches.append(cust)
            demand[cust] = float(rec['values']['demand'])
    fixed_cost = {'S1': 97.65, 'S2': 99.76, 'S3': 100.76, 'S4': 105.32, 'S5': 98.88}
    transportation_cost = {('S1', 'C1'): 150.74, ('S1', 'C2'): 0.02, ('S1', 'C3'): 49.13, ('S1', 'C4'): 2080.15, ('S1', 'C5'): 426.4, ('S2', 'C1'): 233.05, ('S2', 'C2'): 97.73, ('S2', 'C3'): 49.84, ('S2', 'C4'): 1982.39, ('S2', 'C5'): 23.96, ('S3', 'C1'): 55.68, ('S3', 'C2'): 935.61, ('S3', 'C3'): 4.03, ('S3', 'C4'): 73.09, ('S3', 'C5'): 525.32, ('S4', 'C1'): 1483.82, ('S4', 'C2'): 1801.08, ('S4', 'C3'): 112.16, ('S4', 'C4'): 816.05, ('S4', 'C5'): 107.01, ('S5', 'C1'): 1119.47, ('S5', 'C2'): 884.31, ('S5', 'C3'): 0.08, ('S5', 'C4'): 1544.95, ('S5', 'C5'): 543.67}
    for i in suppliers:
        if i not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {i}')
        for j in branches:
            if (i, j) not in transportation_cost:
                raise ValueError(f'Missing transportation cost for supplier {i}, branch {j}')
    for j in branches:
        if j not in demand:
            raise ValueError(f'Missing demand for branch {j}')
    m = Model()
    m.Params.MIPGap = 0.0001
    y_vars = m.addVars(suppliers, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x_vars = m.addVars(suppliers, branches, vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(quicksum((fixed_cost[i] * y_vars[i] for i in suppliers)) + quicksum((transportation_cost[i, j] * x_vars[i, j] for i in suppliers for j in branches)), GRB.MINIMIZE)
    for j in branches:
        m.addConstr(quicksum((x_vars[i, j] for i in suppliers)) == demand[j], name='demand_' + j)
    for i in suppliers:
        for j in branches:
            m.addConstr(x_vars[i, j] <= demand[j] * y_vars[i], name='supplylink_%s_%s' % (i, j))
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for v in m.getVars():
            print(v.VarName, v.X)
    else:
        print(m.Status)
    return m
m = build_and_solve()