LEGACY_OBSERVATION = 'capacity.csv\n{"values": {"ShelfID": "1", "Capacity": "750"}}\n{"values": {"ShelfID": "2", "Capacity": "820"}}\n{"values": {"ShelfID": "3", "Capacity": "570"}}\n{"values": {"ShelfID": "4", "Capacity": "800"}}\n{"values": {"ShelfID": "5", "Capacity": "550"}}\n{"values": {"ShelfID": "6", "Capacity": "900"}}\n{"values": {"ShelfID": "7", "Capacity": "650"}}\n{"values": {"ShelfID": "8", "Capacity": "800"}}\n{"values": {"ShelfID": "9", "Capacity": "850"}}\n{"values": {"ShelfID": "10", "Capacity": "900"}}\n\nproducts.csv\n{"values": {"ProductName": "1", "Value": "55", "Weight": "10"}}\n{"values": {"ProductName": "2", "Value": "75", "Weight": "20"}}\n{"values": {"ProductName": "3", "Value": "65", "Weight": "5"}}\n{"values": {"ProductName": "4", "Value": "60", "Weight": "15"}}\n{"values": {"ProductName": "5", "Value": "80", "Weight": "25"}}\n{"values": {"ProductName": "6", "Value": "90", "Weight": "35"}}\n{"values": {"ProductName": "7", "Value": "40", "Weight": "45"}}\n{"values": {"ProductName": "8", "Value": "100", "Weight": "55"}}\n{"values": {"ProductName": "9", "Value": "55", "Weight": "65"}}\n{"values": {"ProductName": "10", "Value": "75", "Weight": "20"}}\n{"values": {"ProductName": "11", "Value": "110", "Weight": "18"}}\n{"values": {"ProductName": "12", "Value": "50", "Weight": "28"}}\n{"values": {"ProductName": "13", "Value": "60", "Weight": "8"}}\n{"values": {"ProductName": "14", "Value": "120", "Weight": "28"}}\n{"values": {"ProductName": "15", "Value": "70", "Weight": "25"}}\n{"values": {"ProductName": "16", "Value": "110", "Weight": "40"}}\n{"values": {"ProductName": "17", "Value": "50", "Weight": "55"}}\n{"values": {"ProductName": "18", "Value": "60", "Weight": "70"}}\n{"values": {"ProductName": "19", "Value": "120", "Weight": "85"}}\n{"values": {"ProductName": "20", "Value": "100", "Weight": "100"}}'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'ShelfID': '1', 'Capacity': '750'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '2', 'Capacity': '820'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '3', 'Capacity': '570'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '4', 'Capacity': '800'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '5', 'Capacity': '550'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '6', 'Capacity': '900'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '7', 'Capacity': '650'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '8', 'Capacity': '800'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '9', 'Capacity': '850'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '10', 'Capacity': '900'}}, {'source': 'products.csv', 'values': {'ProductName': '1', 'Value': '55', 'Weight': '10'}}, {'source': 'products.csv', 'values': {'ProductName': '2', 'Value': '75', 'Weight': '20'}}, {'source': 'products.csv', 'values': {'ProductName': '3', 'Value': '65', 'Weight': '5'}}, {'source': 'products.csv', 'values': {'ProductName': '4', 'Value': '60', 'Weight': '15'}}, {'source': 'products.csv', 'values': {'ProductName': '5', 'Value': '80', 'Weight': '25'}}, {'source': 'products.csv', 'values': {'ProductName': '6', 'Value': '90', 'Weight': '35'}}, {'source': 'products.csv', 'values': {'ProductName': '7', 'Value': '40', 'Weight': '45'}}, {'source': 'products.csv', 'values': {'ProductName': '8', 'Value': '100', 'Weight': '55'}}, {'source': 'products.csv', 'values': {'ProductName': '9', 'Value': '55', 'Weight': '65'}}, {'source': 'products.csv', 'values': {'ProductName': '10', 'Value': '75', 'Weight': '20'}}, {'source': 'products.csv', 'values': {'ProductName': '11', 'Value': '110', 'Weight': '18'}}, {'source': 'products.csv', 'values': {'ProductName': '12', 'Value': '50', 'Weight': '28'}}, {'source': 'products.csv', 'values': {'ProductName': '13', 'Value': '60', 'Weight': '8'}}, {'source': 'products.csv', 'values': {'ProductName': '14', 'Value': '120', 'Weight': '28'}}, {'source': 'products.csv', 'values': {'ProductName': '15', 'Value': '70', 'Weight': '25'}}, {'source': 'products.csv', 'values': {'ProductName': '16', 'Value': '110', 'Weight': '40'}}, {'source': 'products.csv', 'values': {'ProductName': '17', 'Value': '50', 'Weight': '55'}}, {'source': 'products.csv', 'values': {'ProductName': '18', 'Value': '60', 'Weight': '70'}}, {'source': 'products.csv', 'values': {'ProductName': '19', 'Value': '120', 'Weight': '85'}}, {'source': 'products.csv', 'values': {'ProductName': '20', 'Value': '100', 'Weight': '100'}}]
from gurobipy import Model, GRB

def solve_bigmart_allocation():
    global LEGACY_RECORDS
    shelf_caps = {}
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'capacity.csv':
            sid = rec['values']['ShelfID']
            cap = rec['values']['Capacity']
            if sid in shelf_caps:
                raise ValueError(f'Duplicate shelf id {sid}')
            shelf_caps[sid] = int(cap)
    prod_vals = {}
    prod_wts = {}
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'products.csv':
            pid = rec['values']['ProductName']
            val = rec['values']['Value']
            wt = rec['values']['Weight']
            if pid in prod_vals or pid in prod_wts:
                raise ValueError(f'Duplicate product id {pid}')
            prod_vals[pid] = int(val)
            prod_wts[pid] = int(wt)
    shelf_ids = sorted(shelf_caps.keys(), key=lambda x: int(x))
    prod_ids = sorted(prod_vals.keys(), key=lambda x: int(x))
    if set(prod_vals.keys()) != set(prod_wts.keys()):
        raise ValueError('Mismatch in product value/weight keys')
    for sid in shelf_ids:
        if sid not in shelf_caps:
            raise ValueError(f'Missing capacity for shelf {sid}')
    for pid in prod_ids:
        if pid not in prod_vals or pid not in prod_wts:
            raise ValueError(f'Missing value/weight for product {pid}')
    m = Model()
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(shelf_ids, prod_ids, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(sum((prod_vals[j] * x[i, j] for i in shelf_ids for j in prod_ids)), GRB.MAXIMIZE)
    for i in shelf_ids:
        m.addConstr(sum((prod_wts[j] * x[i, j] for j in prod_ids)) <= shelf_caps[i], name=f'cap_{i}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for i in shelf_ids:
            for j in prod_ids:
                v = x[i, j]
                print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_bigmart_allocation()