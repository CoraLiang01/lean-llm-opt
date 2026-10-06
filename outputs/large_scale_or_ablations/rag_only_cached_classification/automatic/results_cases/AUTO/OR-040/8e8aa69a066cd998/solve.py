LEGACY_OBSERVATION = '{"values": {"Capacity": "4466"}}\n{"values": {"ProductName": "Queens", "Value": "443", "Weight": "104"}}\n{"values": {"ProductName": "Brooklyn", "Value": "522", "Weight": "368"}}\n{"values": {"ProductName": "Manhattan", "Value": "300", "Weight": "483"}}\n{"values": {"ProductName": "Bronx", "Value": "767", "Weight": "165"}}\n{"values": {"ProductName": "Staten Island", "Value": "300", "Weight": "105"}}\n{"values": {"ProductName": "Harlem", "Value": "309", "Weight": "123"}}\n{"values": {"ProductName": "Upper East Side", "Value": "598", "Weight": "131"}}\n{"values": {"ProductName": "Lower Manhattan", "Value": "460", "Weight": "341"}}\n{"values": {"ProductName": "Midtown", "Value": "318", "Weight": "258"}}\n{"values": {"ProductName": "Long Island City", "Value": "126", "Weight": "469"}}\n{"values": {"ProductName": "Williamsburg", "Value": "593", "Weight": "387"}}\n{"values": {"ProductName": "Bushwick", "Value": "871", "Weight": "425"}}\n{"values": {"ProductName": "Flatbush", "Value": "858", "Weight": "482"}}\n{"values": {"ProductName": "Greenpoint", "Value": "321", "Weight": "495"}}\n{"values": {"ProductName": "Park Slope", "Value": "275", "Weight": "305"}}\n{"values": {"ProductName": "Astoria", "Value": "700", "Weight": "377"}}\n{"values": {"ProductName": "Jackson Heights", "Value": "685", "Weight": "318"}}\n{"values": {"ProductName": "Flushing", "Value": "940", "Weight": "56"}}\n{"values": {"ProductName": "Sunnyside", "Value": "522", "Weight": "213"}}\n{"values": {"ProductName": "Ditmars", "Value": "763", "Weight": "472"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Capacity': '4466'}}, {'source': '', 'values': {'ProductName': 'Queens', 'Value': '443', 'Weight': '104'}}, {'source': '', 'values': {'ProductName': 'Brooklyn', 'Value': '522', 'Weight': '368'}}, {'source': '', 'values': {'ProductName': 'Manhattan', 'Value': '300', 'Weight': '483'}}, {'source': '', 'values': {'ProductName': 'Bronx', 'Value': '767', 'Weight': '165'}}, {'source': '', 'values': {'ProductName': 'Staten Island', 'Value': '300', 'Weight': '105'}}, {'source': '', 'values': {'ProductName': 'Harlem', 'Value': '309', 'Weight': '123'}}, {'source': '', 'values': {'ProductName': 'Upper East Side', 'Value': '598', 'Weight': '131'}}, {'source': '', 'values': {'ProductName': 'Lower Manhattan', 'Value': '460', 'Weight': '341'}}, {'source': '', 'values': {'ProductName': 'Midtown', 'Value': '318', 'Weight': '258'}}, {'source': '', 'values': {'ProductName': 'Long Island City', 'Value': '126', 'Weight': '469'}}, {'source': '', 'values': {'ProductName': 'Williamsburg', 'Value': '593', 'Weight': '387'}}, {'source': '', 'values': {'ProductName': 'Bushwick', 'Value': '871', 'Weight': '425'}}, {'source': '', 'values': {'ProductName': 'Flatbush', 'Value': '858', 'Weight': '482'}}, {'source': '', 'values': {'ProductName': 'Greenpoint', 'Value': '321', 'Weight': '495'}}, {'source': '', 'values': {'ProductName': 'Park Slope', 'Value': '275', 'Weight': '305'}}, {'source': '', 'values': {'ProductName': 'Astoria', 'Value': '700', 'Weight': '377'}}, {'source': '', 'values': {'ProductName': 'Jackson Heights', 'Value': '685', 'Weight': '318'}}, {'source': '', 'values': {'ProductName': 'Flushing', 'Value': '940', 'Weight': '56'}}, {'source': '', 'values': {'ProductName': 'Sunnyside', 'Value': '522', 'Weight': '213'}}, {'source': '', 'values': {'ProductName': 'Ditmars', 'Value': '763', 'Weight': '472'}}]
from gurobipy import Model, GRB

def solve_real_estate_optimization():
    global LEGACY_RECORDS
    capacity = None
    products = []
    for rec in LEGACY_RECORDS:
        if rec['source'] == '' and 'Capacity' in rec['values']:
            if capacity is not None:
                raise ValueError('Multiple capacity records found.')
            capacity = int(rec['values']['Capacity'])
        elif rec['source'] == '' and 'ProductName' in rec['values']:
            pname = rec['values']['ProductName']
            try:
                value = int(rec['values']['Value'])
                weight = int(rec['values']['Weight'])
            except Exception:
                raise ValueError(f'Non-integer Value or Weight for {pname}')
            products.append({'ProductName': pname, 'Value': value, 'Weight': weight})
    if capacity is None:
        raise ValueError('No capacity record found.')
    if len(products) == 0:
        raise ValueError('No product records found.')
    for p in products:
        if not all((k in p for k in ('ProductName', 'Value', 'Weight'))):
            raise ValueError(f'Missing fields in product record: {p}')
    product_keys = [p['ProductName'] for p in products]
    value_coeffs = {p['ProductName']: p['Value'] for p in products}
    weight_coeffs = {p['ProductName']: p['Weight'] for p in products}
    if set(value_coeffs.keys()) != set(product_keys):
        raise ValueError('Mismatch in value coefficients and product keys.')
    if set(weight_coeffs.keys()) != set(product_keys):
        raise ValueError('Mismatch in weight coefficients and product keys.')
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(product_keys, vtype=GRB.INTEGER, lb=0, obj=0, name='')
    m.setObjective(sum((value_coeffs[k] * x[k] for k in product_keys)), GRB.MAXIMIZE)
    m.addConstr(sum((weight_coeffs[k] * x[k] for k in product_keys)) <= capacity, name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for k in product_keys:
            print(f'{x[k].VarName} {x[k].X}')
    else:
        print(f'Solver status: {m.Status}')
m = solve_real_estate_optimization()