LEGACY_OBSERVATION = 'capacity.csv\nShelfID,Capacity\n1,500\n2,700\n3,600\n4,800\n5,550\n6,900\n7,650\n8,750\n9,820\n10,570\n\nproducts.csv\nProductName,Value,Weight\n1,50,10\n2,70,20\n3,30,5\n4,60,15\n5,80,25\n6,90,30\n7,40,12\n8,100,35\n9,55,10\n10,75,20\n11,65,18\n12,95,28\n13,45,8\n14,85,22\n15,70,25\n16,110,40\n17,50,14\n18,60,16\n19,120,50\n20,100,30'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'ShelfID': '1', 'Capacity': '500'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '2', 'Capacity': '700'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '3', 'Capacity': '600'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '4', 'Capacity': '800'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '5', 'Capacity': '550'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '6', 'Capacity': '900'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '7', 'Capacity': '650'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '8', 'Capacity': '750'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '9', 'Capacity': '820'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '10', 'Capacity': '570'}}, {'source': 'products.csv', 'values': {'ProductName': '1', 'Value': '50', 'Weight': '10'}}, {'source': 'products.csv', 'values': {'ProductName': '2', 'Value': '70', 'Weight': '20'}}, {'source': 'products.csv', 'values': {'ProductName': '3', 'Value': '30', 'Weight': '5'}}, {'source': 'products.csv', 'values': {'ProductName': '4', 'Value': '60', 'Weight': '15'}}, {'source': 'products.csv', 'values': {'ProductName': '5', 'Value': '80', 'Weight': '25'}}, {'source': 'products.csv', 'values': {'ProductName': '6', 'Value': '90', 'Weight': '30'}}, {'source': 'products.csv', 'values': {'ProductName': '7', 'Value': '40', 'Weight': '12'}}, {'source': 'products.csv', 'values': {'ProductName': '8', 'Value': '100', 'Weight': '35'}}, {'source': 'products.csv', 'values': {'ProductName': '9', 'Value': '55', 'Weight': '10'}}, {'source': 'products.csv', 'values': {'ProductName': '10', 'Value': '75', 'Weight': '20'}}, {'source': 'products.csv', 'values': {'ProductName': '11', 'Value': '65', 'Weight': '18'}}, {'source': 'products.csv', 'values': {'ProductName': '12', 'Value': '95', 'Weight': '28'}}, {'source': 'products.csv', 'values': {'ProductName': '13', 'Value': '45', 'Weight': '8'}}, {'source': 'products.csv', 'values': {'ProductName': '14', 'Value': '85', 'Weight': '22'}}, {'source': 'products.csv', 'values': {'ProductName': '15', 'Value': '70', 'Weight': '25'}}, {'source': 'products.csv', 'values': {'ProductName': '16', 'Value': '110', 'Weight': '40'}}, {'source': 'products.csv', 'values': {'ProductName': '17', 'Value': '50', 'Weight': '14'}}, {'source': 'products.csv', 'values': {'ProductName': '18', 'Value': '60', 'Weight': '16'}}, {'source': 'products.csv', 'values': {'ProductName': '19', 'Value': '120', 'Weight': '50'}}, {'source': 'products.csv', 'values': {'ProductName': '20', 'Value': '100', 'Weight': '30'}}]
from gurobipy import Model, GRB

def solve_bigmart_shelf_allocation():
    global LEGACY_RECORDS
    shelf_caps = []
    shelf_ids = []
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'capacity.csv':
            shelf_ids.append(rec['values']['ShelfID'])
            shelf_caps.append(int(rec['values']['Capacity']))
    if len(shelf_ids) != len(shelf_caps):
        raise ValueError('Mismatch in shelf IDs and capacities')
    n_shelves = len(shelf_ids)
    prod_ids = []
    prod_vals = []
    prod_wgts = []
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'products.csv':
            prod_ids.append(rec['values']['ProductName'])
            prod_vals.append(int(rec['values']['Value']))
            prod_wgts.append(int(rec['values']['Weight']))
    if len(prod_ids) != len(prod_vals) or len(prod_ids) != len(prod_wgts):
        raise ValueError('Mismatch in product data')
    n_prods = len(prod_ids)
    shelf_idx = list(range(n_shelves))
    prod_idx = list(range(n_prods))
    shelfid_by_idx = {i: shelf_ids[i] for i in shelf_idx}
    prodid_by_idx = {j: prod_ids[j] for j in prod_idx}
    if not (len(shelf_caps) == n_shelves and len(prod_vals) == n_prods and (len(prod_wgts) == n_prods)):
        raise ValueError('Coefficient dimension mismatch')
    m = Model()
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(shelf_idx, prod_idx, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(sum((prod_vals[j] * x[i, j] for i in shelf_idx for j in prod_idx)), GRB.MAXIMIZE)
    for i in shelf_idx:
        m.addConstr(sum((prod_wgts[j] * x[i, j] for j in prod_idx)) <= shelf_caps[i], name='cap_%d' % i)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal:', m.ObjVal)
        for i in shelf_idx:
            for j in prod_idx:
                v = x[i, j]
                print(f'x[{shelfid_by_idx[i]}][{prodid_by_idx[j]}] {v.X}')
    else:
        print('Solver status:', m.Status)
m = solve_bigmart_shelf_allocation()