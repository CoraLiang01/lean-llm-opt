LEGACY_OBSERVATION = 'capacity.csv\nSectionID,Capacity\n1,100\n2,150\n3,120\n4,130\n5,90\n6,110\n7,160\n8,140\n\nproducts.csv\nProductName,Value,Weight\n1,10,2\n2,15,3\n3,8,1\n4,12,2\n5,20,4\n6,25,5\n7,5,1\n8,30,6\n9,18,3\n10,22,4'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'SectionID': '1', 'Capacity': '100'}}, {'source': 'capacity.csv', 'values': {'SectionID': '2', 'Capacity': '150'}}, {'source': 'capacity.csv', 'values': {'SectionID': '3', 'Capacity': '120'}}, {'source': 'capacity.csv', 'values': {'SectionID': '4', 'Capacity': '130'}}, {'source': 'capacity.csv', 'values': {'SectionID': '5', 'Capacity': '90'}}, {'source': 'capacity.csv', 'values': {'SectionID': '6', 'Capacity': '110'}}, {'source': 'capacity.csv', 'values': {'SectionID': '7', 'Capacity': '160'}}, {'source': 'capacity.csv', 'values': {'SectionID': '8', 'Capacity': '140'}}, {'source': 'products.csv', 'values': {'ProductName': '1', 'Value': '10', 'Weight': '2'}}, {'source': 'products.csv', 'values': {'ProductName': '2', 'Value': '15', 'Weight': '3'}}, {'source': 'products.csv', 'values': {'ProductName': '3', 'Value': '8', 'Weight': '1'}}, {'source': 'products.csv', 'values': {'ProductName': '4', 'Value': '12', 'Weight': '2'}}, {'source': 'products.csv', 'values': {'ProductName': '5', 'Value': '20', 'Weight': '4'}}, {'source': 'products.csv', 'values': {'ProductName': '6', 'Value': '25', 'Weight': '5'}}, {'source': 'products.csv', 'values': {'ProductName': '7', 'Value': '5', 'Weight': '1'}}, {'source': 'products.csv', 'values': {'ProductName': '8', 'Value': '30', 'Weight': '6'}}, {'source': 'products.csv', 'values': {'ProductName': '9', 'Value': '18', 'Weight': '3'}}, {'source': 'products.csv', 'values': {'ProductName': '10', 'Value': '22', 'Weight': '4'}}]
from gurobipy import Model, GRB

def solve_supermarket_optimization():
    global LEGACY_RECORDS
    capacities = {}
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'capacity.csv':
            sid = int(rec['values']['SectionID'])
            cap = int(rec['values']['Capacity'])
            if sid in capacities:
                raise ValueError(f'Duplicate SectionID {sid} in capacity.csv')
            capacities[sid] = cap
    values = {}
    weights = {}
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'products.csv':
            pid = int(rec['values']['ProductName'])
            val = int(rec['values']['Value'])
            wt = int(rec['values']['Weight'])
            if pid in values or pid in weights:
                raise ValueError(f'Duplicate ProductName {pid} in products.csv')
            values[pid] = val
            weights[pid] = wt
    section_ids = sorted(capacities.keys())
    product_ids = sorted(values.keys())
    if set(weights.keys()) != set(product_ids):
        raise ValueError('Mismatch in product IDs between values and weights')
    if len(section_ids) == 0 or len(product_ids) == 0:
        raise ValueError('No sections or products found')
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(section_ids, product_ids, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(sum((values[j] * x[i, j] for i in section_ids for j in product_ids)), GRB.MAXIMIZE)
    for i in section_ids:
        expr = sum((weights[j] * x[i, j] for j in product_ids))
        m.addConstr(expr <= capacities[i], name='cap_%d' % i)
    for i in section_ids:
        if i not in capacities:
            raise ValueError(f'Missing capacity for section {i}')
    for j in product_ids:
        if j not in values or j not in weights:
            raise ValueError(f'Missing value or weight for product {j}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal:', m.ObjVal)
        for i in section_ids:
            for j in product_ids:
                v = x[i, j]
                print(f'{v.VarName} {v.X}')
    else:
        print('Solver status:', m.Status)
m = solve_supermarket_optimization()