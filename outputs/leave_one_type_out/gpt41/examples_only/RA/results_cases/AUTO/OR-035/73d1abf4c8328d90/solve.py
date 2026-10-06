LEGACY_OBSERVATION = 'capacity.csv\nCapacity\n180\n\nproducts.csv\nProductName,Value,Weight\nBaguette,888,4\nCroissant,134,2\nSourdough,129,4\nRye Bread,370,3\nBrioche,921,2\nFocaccia,765,1\nCiabatta,154,2\nPita,837,1\nBagel,584,3\nEnglish Muffin,365,3'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'Capacity': '180'}}, {'source': 'products.csv', 'values': {'ProductName': 'Baguette', 'Value': '888', 'Weight': '4'}}, {'source': 'products.csv', 'values': {'ProductName': 'Croissant', 'Value': '134', 'Weight': '2'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sourdough', 'Value': '129', 'Weight': '4'}}, {'source': 'products.csv', 'values': {'ProductName': 'Rye Bread', 'Value': '370', 'Weight': '3'}}, {'source': 'products.csv', 'values': {'ProductName': 'Brioche', 'Value': '921', 'Weight': '2'}}, {'source': 'products.csv', 'values': {'ProductName': 'Focaccia', 'Value': '765', 'Weight': '1'}}, {'source': 'products.csv', 'values': {'ProductName': 'Ciabatta', 'Value': '154', 'Weight': '2'}}, {'source': 'products.csv', 'values': {'ProductName': 'Pita', 'Value': '837', 'Weight': '1'}}, {'source': 'products.csv', 'values': {'ProductName': 'Bagel', 'Value': '584', 'Weight': '3'}}, {'source': 'products.csv', 'values': {'ProductName': 'English Muffin', 'Value': '365', 'Weight': '3'}}]
from gurobipy import Model, GRB

def solve_bakery_optimization():
    capacities = {}
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'capacity.csv':
            for k, v in rec['values'].items():
                capacities[k] = int(v)
    if 'Capacity' not in capacities:
        raise ValueError("Missing capacity for 'Capacity' in LEGACY_RECORDS")
    total_capacity = capacities['Capacity']
    products = []
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'products.csv':
            vals = rec['values']
            if not all((k in vals for k in ['ProductName', 'Value', 'Weight'])):
                raise ValueError('Missing product data in LEGACY_RECORDS')
            products.append({'ProductName': vals['ProductName'], 'Value': int(vals['Value']), 'Weight': int(vals['Weight'])})
    if len(products) != 10:
        raise ValueError('Expected 10 products in LEGACY_RECORDS, got %d' % len(products))
    n = len(products)
    product_keys = [p['ProductName'] for p in products]
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(product_keys, vtype=GRB.INTEGER, lb=0, obj=0, name='')
    obj = sum((products[i]['Value'] * x[products[i]['ProductName']] for i in range(n)))
    m.setObjective(obj, GRB.MAXIMIZE)
    cap_expr = sum((products[i]['Weight'] * x[products[i]['ProductName']] for i in range(n)))
    m.addConstr(cap_expr <= total_capacity, name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal', m.ObjVal)
        for i in range(n):
            var = x[products[i]['ProductName']]
            print(var.VarName, var.X)
    else:
        print('Status', m.Status)
m = solve_bakery_optimization()