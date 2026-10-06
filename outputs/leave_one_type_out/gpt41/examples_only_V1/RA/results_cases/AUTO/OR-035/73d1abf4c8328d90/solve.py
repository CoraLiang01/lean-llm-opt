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
        raise ValueError("Missing 'Capacity' in capacity.csv")
    total_capacity = capacities['Capacity']
    products = []
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'products.csv':
            v = rec['values']
            if not all((k in v for k in ['ProductName', 'Value', 'Weight'])):
                raise ValueError('Missing fields in products.csv record')
            products.append({'ProductName': v['ProductName'], 'Value': int(v['Value']), 'Weight': int(v['Weight'])})
    if len(products) != 10:
        raise ValueError('Expected 10 products, got %d' % len(products))
    if any(('Value' not in p or 'Weight' not in p for p in products)):
        raise ValueError('Missing Value or Weight in products')
    product_names = [p['ProductName'] for p in products]
    values = {p['ProductName']: p['Value'] for p in products}
    weights = {p['ProductName']: p['Weight'] for p in products}
    m = Model()
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(product_names, vtype=GRB.INTEGER, lb=0, obj=0, name='')
    m.setObjective(sum((values[name] * x[name] for name in product_names)), GRB.MAXIMIZE)
    m.addConstr(sum((weights[name] * x[name] for name in product_names)) <= total_capacity, name='storage')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal', int(round(m.ObjVal)))
        for name in product_names:
            print(x[name].VarName, int(round(x[name].X)))
    else:
        print('Status', m.Status)
    return m
m = solve_bakery_optimization()