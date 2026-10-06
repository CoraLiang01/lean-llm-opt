LEGACY_OBSERVATION = 'capacity.csv\nCapacity\n1035\n\nproducts.csv\nProductName,Weight,Value\nSpinach,282,49\nShiitake Mushrooms,83,30\nApples,251,30\nCarrots,257,18\nBasil,88,54\nPotatoes,52,27\nGreen Beans,198,91\nBlueberries,203,88\nOranges,87,78\nWatermelons,265,22'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'Capacity': '1035'}}, {'source': 'products.csv', 'values': {'ProductName': 'Spinach', 'Weight': '282', 'Value': '49'}}, {'source': 'products.csv', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '83', 'Value': '30'}}, {'source': 'products.csv', 'values': {'ProductName': 'Apples', 'Weight': '251', 'Value': '30'}}, {'source': 'products.csv', 'values': {'ProductName': 'Carrots', 'Weight': '257', 'Value': '18'}}, {'source': 'products.csv', 'values': {'ProductName': 'Basil', 'Weight': '88', 'Value': '54'}}, {'source': 'products.csv', 'values': {'ProductName': 'Potatoes', 'Weight': '52', 'Value': '27'}}, {'source': 'products.csv', 'values': {'ProductName': 'Green Beans', 'Weight': '198', 'Value': '91'}}, {'source': 'products.csv', 'values': {'ProductName': 'Blueberries', 'Weight': '203', 'Value': '88'}}, {'source': 'products.csv', 'values': {'ProductName': 'Oranges', 'Weight': '87', 'Value': '78'}}, {'source': 'products.csv', 'values': {'ProductName': 'Watermelons', 'Weight': '265', 'Value': '22'}}]
from gurobipy import Model, GRB

def solve_supermarket_knapsack():
    products = []
    capacity = None
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'products.csv':
            products.append({'ProductName': rec['values']['ProductName'], 'Weight': int(rec['values']['Weight']), 'Value': int(rec['values']['Value'])})
        elif rec['source'] == 'capacity.csv':
            if capacity is not None:
                raise ValueError('Multiple capacities found in LEGACY_RECORDS')
            capacity = int(rec['values']['Capacity'])
    if capacity is None:
        raise ValueError('No capacity found in LEGACY_RECORDS')
    if len(products) != 10:
        raise ValueError('Expected 10 products, found %d' % len(products))
    for p in products:
        if not all((k in p for k in ('ProductName', 'Weight', 'Value'))):
            raise ValueError('Missing product fields in LEGACY_RECORDS')
    product_keys = [p['ProductName'] for p in products]
    weights = {p['ProductName']: p['Weight'] for p in products}
    values = {p['ProductName']: p['Value'] for p in products}
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(product_keys, vtype=GRB.INTEGER, lb=0, obj=0, name='')
    m.setObjective(sum((values[k] * x[k] for k in product_keys)), GRB.MAXIMIZE)
    m.addConstr(sum((weights[k] * x[k] for k in product_keys)) <= capacity, name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal', m.ObjVal)
        for k in product_keys:
            print(x[k].VarName, x[k].X)
    else:
        print('Status', m.Status)
    return m
m = solve_supermarket_knapsack()