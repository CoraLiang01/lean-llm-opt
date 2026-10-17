LEGACY_OBSERVATION = '[{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv","values":{"Unnamed: 0":"S1","C1":"1506.22","C2":"70.90000000000001","C3":"8.44","C4":"260.27","C5":"197.47","C6":"71.70999999999999","C7":"61.19"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv","values":{"Unnamed: 0":"S2","C1":"1732.65","C2":"1780.72","C3":"567.4400000000001","C4":"448.68","C5":"29","C6":"1484.91","C7":"963.92"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv","values":{"Unnamed: 0":"S3","C1":"115.66","C2":"100.76","C3":"64.68000000000001","C4":"1324.53","C5":"64.98999999999999","C6":"134.88","C7":"2102.83"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv","values":{"Unnamed: 0":"S4","C1":"1254.78","C2":"1115.63","C3":"52.31","C4":"1036.16","C5":"892.63","C6":"1464.04","C7":"1383.41"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv","values":{"Unnamed: 0":"S5","C1":"42.9","C2":"891.01","C3":"1013.94","C4":"1128.72","C5":"58.91","C6":"42.89","C7":"1570.31"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv","values":{"Unnamed: 0":"S6","C1":"0.7","C2":"139.46","C3":"70.03","C4":"79.15000000000001","C5":"1482","C6":"0.91","C7":"110.46"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv","values":{"Unnamed: 0":"S7","C1":"1732.3","C2":"1780.44","C3":"486.5","C4":"523.74","C5":"522.08","C6":"82.48","C7":"826.41"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '1506.22', 'C2': '70.90000000000001', 'C3': '8.44', 'C4': '260.27', 'C5': '197.47', 'C6': '71.70999999999999', 'C7': '61.19'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '1732.65', 'C2': '1780.72', 'C3': '567.4400000000001', 'C4': '448.68', 'C5': '29', 'C6': '1484.91', 'C7': '963.92'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S3', 'C1': '115.66', 'C2': '100.76', 'C3': '64.68000000000001', 'C4': '1324.53', 'C5': '64.98999999999999', 'C6': '134.88', 'C7': '2102.83'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S4', 'C1': '1254.78', 'C2': '1115.63', 'C3': '52.31', 'C4': '1036.16', 'C5': '892.63', 'C6': '1464.04', 'C7': '1383.41'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S5', 'C1': '42.9', 'C2': '891.01', 'C3': '1013.94', 'C4': '1128.72', 'C5': '58.91', 'C6': '42.89', 'C7': '1570.31'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S6', 'C1': '0.7', 'C2': '139.46', 'C3': '70.03', 'C4': '79.15000000000001', 'C5': '1482', 'C6': '0.91', 'C7': '110.46'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S7', 'C1': '1732.3', 'C2': '1780.44', 'C3': '486.5', 'C4': '523.74', 'C5': '522.08', 'C6': '82.48', 'C7': '826.41'}}]
from gurobipy import Model, GRB, quicksum
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S1', 'C1': '1506.22', 'C2': '70.90000000000001', 'C3': '8.44', 'C4': '260.27', 'C5': '197.47', 'C6': '71.70999999999999', 'C7': '61.19'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S2', 'C1': '1732.65', 'C2': '1780.72', 'C3': '567.4400000000001', 'C4': '448.68', 'C5': '29', 'C6': '1484.91', 'C7': '963.92'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S3', 'C1': '115.66', 'C2': '100.76', 'C3': '64.68000000000001', 'C4': '1324.53', 'C5': '64.98999999999999', 'C6': '134.88', 'C7': '2102.83'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S4', 'C1': '1254.78', 'C2': '1115.63', 'C3': '52.31', 'C4': '1036.16', 'C5': '892.63', 'C6': '1464.04', 'C7': '1383.41'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S5', 'C1': '42.9', 'C2': '891.01', 'C3': '1013.94', 'C4': '1128.72', 'C5': '58.91', 'C6': '42.89', 'C7': '1570.31'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S6', 'C1': '0.7', 'C2': '139.46', 'C3': '70.03', 'C4': '79.15000000000001', 'C5': '1482', 'C6': '0.91', 'C7': '110.46'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv', 'values': {'Unnamed: 0': 'S7', 'C1': '1732.3', 'C2': '1780.44', 'C3': '486.5', 'C4': '523.74', 'C5': '522.08', 'C6': '82.48', 'C7': '826.41'}}]
warehouse_ids = []
customer_ids = []
for rec in LEGACY_RECORDS:
    wh = rec['values']['Unnamed: 0']
    warehouse_ids.append(wh)
    for k in rec['values']:
        if k != 'Unnamed: 0' and k not in customer_ids:
            customer_ids.append(k)
warehouse_ids = list(warehouse_ids)
customer_ids = list(customer_ids)
fixed_costs = {'S1': 102.33, 'S2': 94.92, 'S3': 91.83, 'S4': 98.71, 'S5': 95.73, 'S6': 99.96, 'S7': 98.16}
demands = {'C1': 1083, 'C2': 776, 'C3': 16214, 'C4': 553, 'C5': 17106, 'C6': 594, 'C7': 732}
transportation_costs = {}
for rec in LEGACY_RECORDS:
    i = rec['values']['Unnamed: 0']
    transportation_costs[i] = {}
    for j in customer_ids:
        transportation_costs[i][j] = float(rec['values'][j])
if set(warehouse_ids) != set(fixed_costs.keys()):
    raise ValueError('Mismatch between warehouse_ids and fixed_costs keys')
if set(customer_ids) != set(demands.keys()):
    raise ValueError('Mismatch between customer_ids and demands keys')
for i in warehouse_ids:
    if i not in transportation_costs:
        raise ValueError(f'Missing transportation costs for warehouse {i}')
    if set(transportation_costs[i].keys()) != set(customer_ids):
        raise ValueError(f'Missing transportation costs for warehouse {i} to all customers')

def build_model():
    m = Model()
    m.Params.MIPGap = 0.0001
    y_vars = m.addVars(warehouse_ids, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x_vars = m.addVars(warehouse_ids, customer_ids, vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(quicksum((fixed_costs[i] * y_vars[i] for i in warehouse_ids)) + quicksum((transportation_costs[i][j] * x_vars[i, j] for i in warehouse_ids for j in customer_ids)), GRB.MINIMIZE)
    for j in customer_ids:
        m.addConstr(quicksum((x_vars[i, j] for i in warehouse_ids)) == demands[j], name='')
    for i in warehouse_ids:
        for j in customer_ids:
            m.addConstr(x_vars[i, j] <= demands[j] * y_vars[i], name='')
    return m
m = build_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print(m.Status)