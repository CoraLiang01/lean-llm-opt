LEGACY_OBSERVATION = 'capacity.csv\nSectionID,Capacity\n1,100\n2,150\n3,120\n4,130\n5,90\n6,110\n7,160\n8,140\n\nproducts.csv\nProductName,Value,Weight\n1,10,2\n2,15,3\n3,8,1\n4,12,2\n5,20,4\n6,25,5\n7,5,1\n8,30,6\n9,18,3\n10,22,4'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'SectionID': '1', 'Capacity': '100'}}, {'source': 'capacity.csv', 'values': {'SectionID': '2', 'Capacity': '150'}}, {'source': 'capacity.csv', 'values': {'SectionID': '3', 'Capacity': '120'}}, {'source': 'capacity.csv', 'values': {'SectionID': '4', 'Capacity': '130'}}, {'source': 'capacity.csv', 'values': {'SectionID': '5', 'Capacity': '90'}}, {'source': 'capacity.csv', 'values': {'SectionID': '6', 'Capacity': '110'}}, {'source': 'capacity.csv', 'values': {'SectionID': '7', 'Capacity': '160'}}, {'source': 'capacity.csv', 'values': {'SectionID': '8', 'Capacity': '140'}}, {'source': 'products.csv', 'values': {'ProductName': '1', 'Value': '10', 'Weight': '2'}}, {'source': 'products.csv', 'values': {'ProductName': '2', 'Value': '15', 'Weight': '3'}}, {'source': 'products.csv', 'values': {'ProductName': '3', 'Value': '8', 'Weight': '1'}}, {'source': 'products.csv', 'values': {'ProductName': '4', 'Value': '12', 'Weight': '2'}}, {'source': 'products.csv', 'values': {'ProductName': '5', 'Value': '20', 'Weight': '4'}}, {'source': 'products.csv', 'values': {'ProductName': '6', 'Value': '25', 'Weight': '5'}}, {'source': 'products.csv', 'values': {'ProductName': '7', 'Value': '5', 'Weight': '1'}}, {'source': 'products.csv', 'values': {'ProductName': '8', 'Value': '30', 'Weight': '6'}}, {'source': 'products.csv', 'values': {'ProductName': '9', 'Value': '18', 'Weight': '3'}}, {'source': 'products.csv', 'values': {'ProductName': '10', 'Value': '22', 'Weight': '4'}}]
from gurobipy import Model, GRB

def solve_supermarket_optimization():
    section_records = [r for r in LEGACY_RECORDS if r['source'] == 'capacity.csv']
    product_records = [r for r in LEGACY_RECORDS if r['source'] == 'products.csv']
    section_ids = []
    section_capacities = {}
    for rec in section_records:
        sid = rec['values']['SectionID']
        cap = rec['values']['Capacity']
        if sid in section_capacities:
            raise ValueError(f'Duplicate SectionID {sid} in capacity.csv')
        section_ids.append(sid)
        section_capacities[sid] = float(cap)
    product_ids = []
    product_values = {}
    product_weights = {}
    for rec in product_records:
        pid = rec['values']['ProductName']
        val = rec['values']['Value']
        wt = rec['values']['Weight']
        if pid in product_values or pid in product_weights:
            raise ValueError(f'Duplicate ProductName {pid} in products.csv')
        product_ids.append(pid)
        product_values[pid] = float(val)
        product_weights[pid] = float(wt)
    if len(section_ids) == 0 or len(product_ids) == 0:
        raise ValueError('No sections or products found in LEGACY_RECORDS')
    for sid in section_ids:
        if sid not in section_capacities:
            raise ValueError(f'Missing capacity for SectionID {sid}')
    for pid in product_ids:
        if pid not in product_values or pid not in product_weights:
            raise ValueError(f'Missing value or weight for ProductName {pid}')
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(section_ids, product_ids, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(sum((product_values[pid] * x[sid, pid] for sid in section_ids for pid in product_ids)), GRB.MAXIMIZE)
    for sid in section_ids:
        m.addConstr(sum((product_weights[pid] * x[sid, pid] for pid in product_ids)) <= section_capacities[sid], name=f'cap_{sid}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for sid in section_ids:
            for pid in product_ids:
                var = x[sid, pid]
                print(f'{var.VarName} {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_supermarket_optimization()