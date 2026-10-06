LEGACY_OBSERVATION = 'capacity.csv\nCapacity\n875\n\nproducts.csv\nProductName,Weight,Value\nSpinach,230,64\nShiitake Mushrooms,637,75\nApples,773,68\nCarrots,653,11\nBasil,755,91\nPotatoes,670,31\nGreen Beans,505,90\nBlueberries,821,56\nOranges,83,10\nWatermelons,249,24'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'Capacity': '875'}}, {'source': 'products.csv', 'values': {'ProductName': 'Spinach', 'Weight': '230', 'Value': '64'}}, {'source': 'products.csv', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '637', 'Value': '75'}}, {'source': 'products.csv', 'values': {'ProductName': 'Apples', 'Weight': '773', 'Value': '68'}}, {'source': 'products.csv', 'values': {'ProductName': 'Carrots', 'Weight': '653', 'Value': '11'}}, {'source': 'products.csv', 'values': {'ProductName': 'Basil', 'Weight': '755', 'Value': '91'}}, {'source': 'products.csv', 'values': {'ProductName': 'Potatoes', 'Weight': '670', 'Value': '31'}}, {'source': 'products.csv', 'values': {'ProductName': 'Green Beans', 'Weight': '505', 'Value': '90'}}, {'source': 'products.csv', 'values': {'ProductName': 'Blueberries', 'Weight': '821', 'Value': '56'}}, {'source': 'products.csv', 'values': {'ProductName': 'Oranges', 'Weight': '83', 'Value': '10'}}, {'source': 'products.csv', 'values': {'ProductName': 'Watermelons', 'Weight': '249', 'Value': '24'}}]
from gurobipy import Model, GRB

def solve_supermarket_optimization():
    products = []
    weights = []
    values = []
    product_names = []
    capacity = None
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'capacity.csv':
            if capacity is not None:
                raise ValueError('Multiple capacities found in LEGACY_RECORDS')
            if 'Capacity' not in rec['values']:
                raise ValueError("Missing 'Capacity' in capacity.csv record")
            capacity = int(rec['values']['Capacity'])
        elif rec['source'] == 'products.csv':
            vals = rec['values']
            if not all((k in vals for k in ['ProductName', 'Weight', 'Value'])):
                raise ValueError('Missing fields in products.csv record')
            product_names.append(vals['ProductName'])
            weights.append(int(vals['Weight']))
            values.append(int(vals['Value']))
            products.append(len(products))
    if capacity is None:
        raise ValueError('No capacity found in LEGACY_RECORDS')
    if len(products) == 0:
        raise ValueError('No products found in LEGACY_RECORDS')
    if not len(products) == len(weights) == len(values) == len(product_names):
        raise ValueError('Inconsistent product data in LEGACY_RECORDS')
    n = len(products)
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(products, vtype=GRB.INTEGER, lb=0, name='')
    obj = sum((values[i] * x[i] for i in products))
    m.setObjective(obj, GRB.MAXIMIZE)
    m.addConstr(sum((weights[i] * x[i] for i in products)) <= capacity, name='stock_cap')
    if any((w is None for w in weights)) or any((v is None for v in values)):
        raise ValueError('Missing weights or values for some products')
    if len(weights) != n or len(values) != n:
        raise ValueError('Mismatch in product data lengths')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal:', m.ObjVal)
        for i in products:
            print(f'x[{product_names[i]}]:', x[i].X)
    else:
        print('Solver status:', m.Status)
m = solve_supermarket_optimization()