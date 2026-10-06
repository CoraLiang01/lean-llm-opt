LEGACY_OBSERVATION = 'capacity.csv\nCapacity\n4466\n\nproducts.csv\nProductName,Value,Weight\nQueens,443,104\nBrooklyn,522,368\nManhattan,300,483\nBronx,767,165\nStaten Island,300,105\nHarlem,309,123\nUpper East Side,598,131\nLower Manhattan,460,341\nMidtown,318,258\nLong Island City,126,469\nWilliamsburg,593,387\nBushwick,871,425\nFlatbush,858,482\nGreenpoint,321,495\nPark Slope,275,305\nAstoria,700,377\nJackson Heights,685,318\nFlushing,940,56\nSunnyside,522,213\nDitmars,763,472'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'Capacity': '4466'}}, {'source': 'products.csv', 'values': {'ProductName': 'Queens', 'Value': '443', 'Weight': '104'}}, {'source': 'products.csv', 'values': {'ProductName': 'Brooklyn', 'Value': '522', 'Weight': '368'}}, {'source': 'products.csv', 'values': {'ProductName': 'Manhattan', 'Value': '300', 'Weight': '483'}}, {'source': 'products.csv', 'values': {'ProductName': 'Bronx', 'Value': '767', 'Weight': '165'}}, {'source': 'products.csv', 'values': {'ProductName': 'Staten Island', 'Value': '300', 'Weight': '105'}}, {'source': 'products.csv', 'values': {'ProductName': 'Harlem', 'Value': '309', 'Weight': '123'}}, {'source': 'products.csv', 'values': {'ProductName': 'Upper East Side', 'Value': '598', 'Weight': '131'}}, {'source': 'products.csv', 'values': {'ProductName': 'Lower Manhattan', 'Value': '460', 'Weight': '341'}}, {'source': 'products.csv', 'values': {'ProductName': 'Midtown', 'Value': '318', 'Weight': '258'}}, {'source': 'products.csv', 'values': {'ProductName': 'Long Island City', 'Value': '126', 'Weight': '469'}}, {'source': 'products.csv', 'values': {'ProductName': 'Williamsburg', 'Value': '593', 'Weight': '387'}}, {'source': 'products.csv', 'values': {'ProductName': 'Bushwick', 'Value': '871', 'Weight': '425'}}, {'source': 'products.csv', 'values': {'ProductName': 'Flatbush', 'Value': '858', 'Weight': '482'}}, {'source': 'products.csv', 'values': {'ProductName': 'Greenpoint', 'Value': '321', 'Weight': '495'}}, {'source': 'products.csv', 'values': {'ProductName': 'Park Slope', 'Value': '275', 'Weight': '305'}}, {'source': 'products.csv', 'values': {'ProductName': 'Astoria', 'Value': '700', 'Weight': '377'}}, {'source': 'products.csv', 'values': {'ProductName': 'Jackson Heights', 'Value': '685', 'Weight': '318'}}, {'source': 'products.csv', 'values': {'ProductName': 'Flushing', 'Value': '940', 'Weight': '56'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sunnyside', 'Value': '522', 'Weight': '213'}}, {'source': 'products.csv', 'values': {'ProductName': 'Ditmars', 'Value': '763', 'Weight': '472'}}]
from gurobipy import Model, GRB

def solve_real_estate_knapsack(LEGACY_RECORDS):
    capacities = [int(r['values']['Capacity']) for r in LEGACY_RECORDS if r['source'] == 'capacity.csv']
    if not capacities:
        raise ValueError('No capacity found in LEGACY_RECORDS')
    if len(capacities) > 1:
        raise ValueError('Multiple capacities found in LEGACY_RECORDS')
    capacity = capacities[0]
    products = [r for r in LEGACY_RECORDS if r['source'] == 'products.csv']
    if not products:
        raise ValueError('No products found in LEGACY_RECORDS')
    for p in products:
        for k in ['ProductName', 'Value', 'Weight']:
            if k not in p['values']:
                raise ValueError(f'Missing {k} in product {p}')
    area_keys = []
    values = {}
    weights = {}
    for p in products:
        name = p['values']['ProductName']
        area_keys.append(name)
        try:
            values[name] = int(p['values']['Value'])
            weights[name] = int(p['values']['Weight'])
        except Exception as e:
            raise ValueError(f'Invalid Value/Weight for {name}: {e}')
    if set(values.keys()) != set(area_keys) or set(weights.keys()) != set(area_keys):
        raise ValueError('Mismatch in area keys and coefficients')
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(area_keys, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(sum((values[a] * x[a] for a in area_keys)), GRB.MAXIMIZE)
    m.addConstr(sum((weights[a] * x[a] for a in area_keys)) <= capacity, name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for a in area_keys:
            print(f'{x[a].VarName} {x[a].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_real_estate_knapsack(LEGACY_RECORDS)