LEGACY_OBSERVATION = 'capacity.csv\nCapacity\n586\n\nproducts.csv\nProductName,Value,Weight\nQueens,469,954\nBrooklyn,290,650\nManhattan,236,961\nBronx,235,950\nStaten Island,745,379\nHarlem,684,776\nUpper East Side,444,381\nLower Manhattan,172,808\nMidtown,1000,937\nLong Island City,336,608\nWilliamsburg,546,912\nBushwick,535,391\nFlatbush,539,465\nGreenpoint,831,490\nPark Slope,139,918\nAstoria,432,787\nJackson Heights,627,347\nFlushing,629,274\nSunnyside,292,642\nDitmars,978,130'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'Capacity': '586'}}, {'source': 'products.csv', 'values': {'ProductName': 'Queens', 'Value': '469', 'Weight': '954'}}, {'source': 'products.csv', 'values': {'ProductName': 'Brooklyn', 'Value': '290', 'Weight': '650'}}, {'source': 'products.csv', 'values': {'ProductName': 'Manhattan', 'Value': '236', 'Weight': '961'}}, {'source': 'products.csv', 'values': {'ProductName': 'Bronx', 'Value': '235', 'Weight': '950'}}, {'source': 'products.csv', 'values': {'ProductName': 'Staten Island', 'Value': '745', 'Weight': '379'}}, {'source': 'products.csv', 'values': {'ProductName': 'Harlem', 'Value': '684', 'Weight': '776'}}, {'source': 'products.csv', 'values': {'ProductName': 'Upper East Side', 'Value': '444', 'Weight': '381'}}, {'source': 'products.csv', 'values': {'ProductName': 'Lower Manhattan', 'Value': '172', 'Weight': '808'}}, {'source': 'products.csv', 'values': {'ProductName': 'Midtown', 'Value': '1000', 'Weight': '937'}}, {'source': 'products.csv', 'values': {'ProductName': 'Long Island City', 'Value': '336', 'Weight': '608'}}, {'source': 'products.csv', 'values': {'ProductName': 'Williamsburg', 'Value': '546', 'Weight': '912'}}, {'source': 'products.csv', 'values': {'ProductName': 'Bushwick', 'Value': '535', 'Weight': '391'}}, {'source': 'products.csv', 'values': {'ProductName': 'Flatbush', 'Value': '539', 'Weight': '465'}}, {'source': 'products.csv', 'values': {'ProductName': 'Greenpoint', 'Value': '831', 'Weight': '490'}}, {'source': 'products.csv', 'values': {'ProductName': 'Park Slope', 'Value': '139', 'Weight': '918'}}, {'source': 'products.csv', 'values': {'ProductName': 'Astoria', 'Value': '432', 'Weight': '787'}}, {'source': 'products.csv', 'values': {'ProductName': 'Jackson Heights', 'Value': '627', 'Weight': '347'}}, {'source': 'products.csv', 'values': {'ProductName': 'Flushing', 'Value': '629', 'Weight': '274'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sunnyside', 'Value': '292', 'Weight': '642'}}, {'source': 'products.csv', 'values': {'ProductName': 'Ditmars', 'Value': '978', 'Weight': '130'}}]
from gurobipy import Model, GRB

def solve_property_development(LEGACY_RECORDS):
    capacities = [r for r in LEGACY_RECORDS if r['source'] == 'capacity.csv']
    if not capacities or 'Capacity' not in capacities[0]['values']:
        raise ValueError('Missing capacity in LEGACY_RECORDS')
    capacity = int(capacities[0]['values']['Capacity'])
    products = [r for r in LEGACY_RECORDS if r['source'] == 'products.csv']
    if len(products) != 20:
        raise ValueError('Expected 20 products in LEGACY_RECORDS')
    area_keys = []
    values = {}
    weights = {}
    for r in products:
        name = r['values']['ProductName']
        area_keys.append(name)
        try:
            values[name] = int(r['values']['Value'])
            weights[name] = int(r['values']['Weight'])
        except Exception as e:
            raise ValueError(f'Invalid Value/Weight for {name}: {e}')
    if set(values.keys()) != set(area_keys) or set(weights.keys()) != set(area_keys):
        raise ValueError('Mismatch in area keys and coefficients')
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(area_keys, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(sum((values[area] * x[area] for area in area_keys)), GRB.MAXIMIZE)
    m.addConstr(sum((weights[area] * x[area] for area in area_keys)) <= capacity, name='capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for area in area_keys:
            print(f'{x[area].VarName}: {x[area].X}')
    else:
        print(f'Solver status: {m.Status}')
m = solve_property_development(LEGACY_RECORDS)