LEGACY_OBSERVATION = 'capacity.csv\nCapacity\n4466\n\nproducts.csv\nProductName,Value,Weight\nQueens,443,104\nBrooklyn,522,368\nManhattan,300,483\nBronx,767,165\nStaten Island,300,105\nHarlem,309,123\nUpper East Side,598,131\nLower Manhattan,460,341\nMidtown,318,258\nLong Island City,126,469\nWilliamsburg,593,387\nBushwick,871,425\nFlatbush,858,482\nGreenpoint,321,495\nPark Slope,275,305\nAstoria,700,377\nJackson Heights,685,318\nFlushing,940,56\nSunnyside,522,213\nDitmars,763,472'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'Capacity': '4466'}}, {'source': 'products.csv', 'values': {'ProductName': 'Queens', 'Value': '443', 'Weight': '104'}}, {'source': 'products.csv', 'values': {'ProductName': 'Brooklyn', 'Value': '522', 'Weight': '368'}}, {'source': 'products.csv', 'values': {'ProductName': 'Manhattan', 'Value': '300', 'Weight': '483'}}, {'source': 'products.csv', 'values': {'ProductName': 'Bronx', 'Value': '767', 'Weight': '165'}}, {'source': 'products.csv', 'values': {'ProductName': 'Staten Island', 'Value': '300', 'Weight': '105'}}, {'source': 'products.csv', 'values': {'ProductName': 'Harlem', 'Value': '309', 'Weight': '123'}}, {'source': 'products.csv', 'values': {'ProductName': 'Upper East Side', 'Value': '598', 'Weight': '131'}}, {'source': 'products.csv', 'values': {'ProductName': 'Lower Manhattan', 'Value': '460', 'Weight': '341'}}, {'source': 'products.csv', 'values': {'ProductName': 'Midtown', 'Value': '318', 'Weight': '258'}}, {'source': 'products.csv', 'values': {'ProductName': 'Long Island City', 'Value': '126', 'Weight': '469'}}, {'source': 'products.csv', 'values': {'ProductName': 'Williamsburg', 'Value': '593', 'Weight': '387'}}, {'source': 'products.csv', 'values': {'ProductName': 'Bushwick', 'Value': '871', 'Weight': '425'}}, {'source': 'products.csv', 'values': {'ProductName': 'Flatbush', 'Value': '858', 'Weight': '482'}}, {'source': 'products.csv', 'values': {'ProductName': 'Greenpoint', 'Value': '321', 'Weight': '495'}}, {'source': 'products.csv', 'values': {'ProductName': 'Park Slope', 'Value': '275', 'Weight': '305'}}, {'source': 'products.csv', 'values': {'ProductName': 'Astoria', 'Value': '700', 'Weight': '377'}}, {'source': 'products.csv', 'values': {'ProductName': 'Jackson Heights', 'Value': '685', 'Weight': '318'}}, {'source': 'products.csv', 'values': {'ProductName': 'Flushing', 'Value': '940', 'Weight': '56'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sunnyside', 'Value': '522', 'Weight': '213'}}, {'source': 'products.csv', 'values': {'ProductName': 'Ditmars', 'Value': '763', 'Weight': '472'}}]
from gurobipy import Model, GRB

def solve_developer_problem():
    global LEGACY_RECORDS
    products = [rec for rec in LEGACY_RECORDS if rec['source'] == 'products.csv']
    capacity_recs = [rec for rec in LEGACY_RECORDS if rec['source'] == 'capacity.csv']
    if not products:
        raise ValueError('No products found in LEGACY_RECORDS')
    if not capacity_recs:
        raise ValueError('No capacity found in LEGACY_RECORDS')
    if len(capacity_recs) != 1:
        raise ValueError('Expected exactly one capacity record')
    area_names = []
    values = {}
    weights = {}
    for prod in products:
        name = prod['values']['ProductName']
        val = int(prod['values']['Value'])
        wt = int(prod['values']['Weight'])
        area_names.append(name)
        values[name] = val
        weights[name] = wt
    for name in area_names:
        if name not in values or name not in weights:
            raise ValueError(f'Missing value or weight for area {name}')
    total_capacity = int(capacity_recs[0]['values']['Capacity'])
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(area_names, vtype=GRB.INTEGER, lb=0, obj=0, name='')
    m.setObjective(sum((values[area] * x[area] for area in area_names)), GRB.MAXIMIZE)
    m.addConstr(sum((weights[area] * x[area] for area in area_names)) <= total_capacity, name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for area in area_names:
            print(f'{x[area].VarName} {x[area].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_developer_problem()