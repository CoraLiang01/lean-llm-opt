LEGACY_OBSERVATION = 'capacity.csv\nCapacity\n586\n\nproducts.csv\nProductName,Value,Weight\nQueens,469,954\nBrooklyn,290,650\nManhattan,236,961\nBronx,235,950\nStaten Island,745,379\nHarlem,684,776\nUpper East Side,444,381\nLower Manhattan,172,808\nMidtown,1000,937\nLong Island City,336,608\nWilliamsburg,546,912\nBushwick,535,391\nFlatbush,539,465\nGreenpoint,831,490\nPark Slope,139,918\nAstoria,432,787\nJackson Heights,627,347\nFlushing,629,274\nSunnyside,292,642\nDitmars,978,130'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'Capacity': '586'}}, {'source': 'products.csv', 'values': {'ProductName': 'Queens', 'Value': '469', 'Weight': '954'}}, {'source': 'products.csv', 'values': {'ProductName': 'Brooklyn', 'Value': '290', 'Weight': '650'}}, {'source': 'products.csv', 'values': {'ProductName': 'Manhattan', 'Value': '236', 'Weight': '961'}}, {'source': 'products.csv', 'values': {'ProductName': 'Bronx', 'Value': '235', 'Weight': '950'}}, {'source': 'products.csv', 'values': {'ProductName': 'Staten Island', 'Value': '745', 'Weight': '379'}}, {'source': 'products.csv', 'values': {'ProductName': 'Harlem', 'Value': '684', 'Weight': '776'}}, {'source': 'products.csv', 'values': {'ProductName': 'Upper East Side', 'Value': '444', 'Weight': '381'}}, {'source': 'products.csv', 'values': {'ProductName': 'Lower Manhattan', 'Value': '172', 'Weight': '808'}}, {'source': 'products.csv', 'values': {'ProductName': 'Midtown', 'Value': '1000', 'Weight': '937'}}, {'source': 'products.csv', 'values': {'ProductName': 'Long Island City', 'Value': '336', 'Weight': '608'}}, {'source': 'products.csv', 'values': {'ProductName': 'Williamsburg', 'Value': '546', 'Weight': '912'}}, {'source': 'products.csv', 'values': {'ProductName': 'Bushwick', 'Value': '535', 'Weight': '391'}}, {'source': 'products.csv', 'values': {'ProductName': 'Flatbush', 'Value': '539', 'Weight': '465'}}, {'source': 'products.csv', 'values': {'ProductName': 'Greenpoint', 'Value': '831', 'Weight': '490'}}, {'source': 'products.csv', 'values': {'ProductName': 'Park Slope', 'Value': '139', 'Weight': '918'}}, {'source': 'products.csv', 'values': {'ProductName': 'Astoria', 'Value': '432', 'Weight': '787'}}, {'source': 'products.csv', 'values': {'ProductName': 'Jackson Heights', 'Value': '627', 'Weight': '347'}}, {'source': 'products.csv', 'values': {'ProductName': 'Flushing', 'Value': '629', 'Weight': '274'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sunnyside', 'Value': '292', 'Weight': '642'}}, {'source': 'products.csv', 'values': {'ProductName': 'Ditmars', 'Value': '978', 'Weight': '130'}}]
from gurobipy import Model, GRB
products = []
capacity = None
for rec in LEGACY_RECORDS:
    if rec['source'] == 'products.csv':
        v = rec['values']
        products.append({'ProductName': v['ProductName'], 'Value': int(v['Value']), 'Weight': int(v['Weight'])})
    elif rec['source'] == 'capacity.csv':
        capacity = int(rec['values']['Capacity'])
if capacity is None:
    raise ValueError('Missing capacity from LEGACY_RECORDS')
if len(products) == 0:
    raise ValueError('No products found in LEGACY_RECORDS')
for p in products:
    if not all((k in p for k in ('ProductName', 'Value', 'Weight'))):
        raise ValueError(f'Missing fields in product: {p}')
area_keys = [p['ProductName'] for p in products]
values = {p['ProductName']: p['Value'] for p in products}
weights = {p['ProductName']: p['Weight'] for p in products}
m = Model()
x = m.addVars(area_keys, vtype=GRB.INTEGER, lb=0, obj=0, name='')
m.setObjective(sum((values[k] * x[k] for k in area_keys)), GRB.MAXIMIZE)
m.addConstr(sum((weights[k] * x[k] for k in area_keys)) <= capacity, name='cap')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('ObjVal', m.ObjVal)
    for k in area_keys:
        print(x[k].VarName, x[k].X)
else:
    print('Status', m.Status)