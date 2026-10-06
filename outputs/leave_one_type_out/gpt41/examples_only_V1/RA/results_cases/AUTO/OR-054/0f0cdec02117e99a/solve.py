LEGACY_OBSERVATION = 'capacity.csv\nShelfID,Capacity\n1,750\n2,820\n3,570\n4,800\n5,550\n6,900\n7,650\n8,800\n9,850\n10,900\n\nproducts.csv\nProductName,Value,Weight\n1,55,10\n2,75,20\n3,65,5\n4,60,15\n5,80,25\n6,90,35\n7,40,45\n8,100,55\n9,55,65\n10,75,20\n11,110,18\n12,50,28\n13,60,8\n14,120,28\n15,70,25\n16,110,40\n17,50,55\n18,60,70\n19,120,85\n20,100,100'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'ShelfID': '1', 'Capacity': '750'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '2', 'Capacity': '820'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '3', 'Capacity': '570'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '4', 'Capacity': '800'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '5', 'Capacity': '550'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '6', 'Capacity': '900'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '7', 'Capacity': '650'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '8', 'Capacity': '800'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '9', 'Capacity': '850'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '10', 'Capacity': '900'}}, {'source': 'products.csv', 'values': {'ProductName': '1', 'Value': '55', 'Weight': '10'}}, {'source': 'products.csv', 'values': {'ProductName': '2', 'Value': '75', 'Weight': '20'}}, {'source': 'products.csv', 'values': {'ProductName': '3', 'Value': '65', 'Weight': '5'}}, {'source': 'products.csv', 'values': {'ProductName': '4', 'Value': '60', 'Weight': '15'}}, {'source': 'products.csv', 'values': {'ProductName': '5', 'Value': '80', 'Weight': '25'}}, {'source': 'products.csv', 'values': {'ProductName': '6', 'Value': '90', 'Weight': '35'}}, {'source': 'products.csv', 'values': {'ProductName': '7', 'Value': '40', 'Weight': '45'}}, {'source': 'products.csv', 'values': {'ProductName': '8', 'Value': '100', 'Weight': '55'}}, {'source': 'products.csv', 'values': {'ProductName': '9', 'Value': '55', 'Weight': '65'}}, {'source': 'products.csv', 'values': {'ProductName': '10', 'Value': '75', 'Weight': '20'}}, {'source': 'products.csv', 'values': {'ProductName': '11', 'Value': '110', 'Weight': '18'}}, {'source': 'products.csv', 'values': {'ProductName': '12', 'Value': '50', 'Weight': '28'}}, {'source': 'products.csv', 'values': {'ProductName': '13', 'Value': '60', 'Weight': '8'}}, {'source': 'products.csv', 'values': {'ProductName': '14', 'Value': '120', 'Weight': '28'}}, {'source': 'products.csv', 'values': {'ProductName': '15', 'Value': '70', 'Weight': '25'}}, {'source': 'products.csv', 'values': {'ProductName': '16', 'Value': '110', 'Weight': '40'}}, {'source': 'products.csv', 'values': {'ProductName': '17', 'Value': '50', 'Weight': '55'}}, {'source': 'products.csv', 'values': {'ProductName': '18', 'Value': '60', 'Weight': '70'}}, {'source': 'products.csv', 'values': {'ProductName': '19', 'Value': '120', 'Weight': '85'}}, {'source': 'products.csv', 'values': {'ProductName': '20', 'Value': '100', 'Weight': '100'}}]
from gurobipy import Model, GRB

def solve_bigmart_allocation():
    shelves = []
    shelf_capacities = {}
    products = []
    product_values = {}
    product_weights = {}
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'capacity.csv':
            shelf_id = rec['values']['ShelfID']
            cap = int(rec['values']['Capacity'])
            shelves.append(shelf_id)
            shelf_capacities[shelf_id] = cap
        elif rec['source'] == 'products.csv':
            prod_id = rec['values']['ProductName']
            val = int(rec['values']['Value'])
            wt = int(rec['values']['Weight'])
            products.append(prod_id)
            product_values[prod_id] = val
            product_weights[prod_id] = wt
    shelves = sorted(shelves, key=lambda x: int(x))
    products = sorted(products, key=lambda x: int(x))
    if len(shelves) != 10:
        raise ValueError('Expected 10 shelves, got %d' % len(shelves))
    if len(products) != 20:
        raise ValueError('Expected 20 products, got %d' % len(products))
    for s in shelves:
        if s not in shelf_capacities:
            raise ValueError(f'Missing capacity for shelf {s}')
    for p in products:
        if p not in product_values or p not in product_weights:
            raise ValueError(f'Missing value/weight for product {p}')
    m = Model()
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(shelves, products, vtype=GRB.INTEGER, lb=0, keyformat=None, name='')
    m.setObjective(sum((product_values[j] * x[i, j] for i in shelves for j in products)), GRB.MAXIMIZE)
    for i in shelves:
        m.addConstr(sum((product_weights[j] * x[i, j] for j in products)) <= shelf_capacities[i], name='cap_' + i)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal', m.ObjVal)
        for i in shelves:
            for j in products:
                v = x[i, j]
                print(v.VarName, v.X)
    else:
        print('Solver status:', m.Status)
m = solve_bigmart_allocation()