LEGACY_OBSERVATION = 'capacity.csv\nDisplayID,Capacity\n1,356\n2,478\n3,305\n4,291\n5,168\n6,449\n7,139\n8,383\n9,472\n10,288\n11,320\n12,250\n13,402\n14,293\n\nproducts.csv\nProductName,Value,Weight\nSpeedboat,69978,18\nFishing Boat,54011,42\nCatamaran,36352,49\nYacht,51521,42\nSailboat,50415,41\nKayak,76109,48\nCanoe,50462,22\nHouseboat,28989,29\nPontoon,23318,45\nJet Ski,26142,14\nRowboat,42040,38\nHovercraft,85961,47\nCabin Cruiser,50142,45\nWakeboard Boat,48478,28\nDinghy,60953,24\nTrawler,95265,39\nPaddle Boat,22839,32\nSubmarine,90957,36\nRIB,84652,14\nSkiff,78991,16'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'DisplayID': '1', 'Capacity': '356'}}, {'source': 'capacity.csv', 'values': {'DisplayID': '2', 'Capacity': '478'}}, {'source': 'capacity.csv', 'values': {'DisplayID': '3', 'Capacity': '305'}}, {'source': 'capacity.csv', 'values': {'DisplayID': '4', 'Capacity': '291'}}, {'source': 'capacity.csv', 'values': {'DisplayID': '5', 'Capacity': '168'}}, {'source': 'capacity.csv', 'values': {'DisplayID': '6', 'Capacity': '449'}}, {'source': 'capacity.csv', 'values': {'DisplayID': '7', 'Capacity': '139'}}, {'source': 'capacity.csv', 'values': {'DisplayID': '8', 'Capacity': '383'}}, {'source': 'capacity.csv', 'values': {'DisplayID': '9', 'Capacity': '472'}}, {'source': 'capacity.csv', 'values': {'DisplayID': '10', 'Capacity': '288'}}, {'source': 'capacity.csv', 'values': {'DisplayID': '11', 'Capacity': '320'}}, {'source': 'capacity.csv', 'values': {'DisplayID': '12', 'Capacity': '250'}}, {'source': 'capacity.csv', 'values': {'DisplayID': '13', 'Capacity': '402'}}, {'source': 'capacity.csv', 'values': {'DisplayID': '14', 'Capacity': '293'}}, {'source': 'products.csv', 'values': {'ProductName': 'Speedboat', 'Value': '69978', 'Weight': '18'}}, {'source': 'products.csv', 'values': {'ProductName': 'Fishing Boat', 'Value': '54011', 'Weight': '42'}}, {'source': 'products.csv', 'values': {'ProductName': 'Catamaran', 'Value': '36352', 'Weight': '49'}}, {'source': 'products.csv', 'values': {'ProductName': 'Yacht', 'Value': '51521', 'Weight': '42'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sailboat', 'Value': '50415', 'Weight': '41'}}, {'source': 'products.csv', 'values': {'ProductName': 'Kayak', 'Value': '76109', 'Weight': '48'}}, {'source': 'products.csv', 'values': {'ProductName': 'Canoe', 'Value': '50462', 'Weight': '22'}}, {'source': 'products.csv', 'values': {'ProductName': 'Houseboat', 'Value': '28989', 'Weight': '29'}}, {'source': 'products.csv', 'values': {'ProductName': 'Pontoon', 'Value': '23318', 'Weight': '45'}}, {'source': 'products.csv', 'values': {'ProductName': 'Jet Ski', 'Value': '26142', 'Weight': '14'}}, {'source': 'products.csv', 'values': {'ProductName': 'Rowboat', 'Value': '42040', 'Weight': '38'}}, {'source': 'products.csv', 'values': {'ProductName': 'Hovercraft', 'Value': '85961', 'Weight': '47'}}, {'source': 'products.csv', 'values': {'ProductName': 'Cabin Cruiser', 'Value': '50142', 'Weight': '45'}}, {'source': 'products.csv', 'values': {'ProductName': 'Wakeboard Boat', 'Value': '48478', 'Weight': '28'}}, {'source': 'products.csv', 'values': {'ProductName': 'Dinghy', 'Value': '60953', 'Weight': '24'}}, {'source': 'products.csv', 'values': {'ProductName': 'Trawler', 'Value': '95265', 'Weight': '39'}}, {'source': 'products.csv', 'values': {'ProductName': 'Paddle Boat', 'Value': '22839', 'Weight': '32'}}, {'source': 'products.csv', 'values': {'ProductName': 'Submarine', 'Value': '90957', 'Weight': '36'}}, {'source': 'products.csv', 'values': {'ProductName': 'RIB', 'Value': '84652', 'Weight': '14'}}, {'source': 'products.csv', 'values': {'ProductName': 'Skiff', 'Value': '78991', 'Weight': '16'}}]
from gurobipy import Model, GRB

def solve_boat_display(LEGACY_RECORDS):
    display_areas = []
    capacities = {}
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'capacity.csv':
            did = rec['values']['DisplayID']
            cap = int(rec['values']['Capacity'])
            display_areas.append(did)
            capacities[did] = cap
    products = []
    values = {}
    weights = {}
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'products.csv':
            pname = rec['values']['ProductName']
            val = int(rec['values']['Value'])
            wt = int(rec['values']['Weight'])
            products.append(pname)
            values[pname] = val
            weights[pname] = wt
    if len(display_areas) == 0 or len(products) == 0:
        raise ValueError('Missing display areas or products.')
    for did in display_areas:
        if did not in capacities:
            raise ValueError(f'Missing capacity for display area {did}')
    for pname in products:
        if pname not in values or pname not in weights:
            raise ValueError(f'Missing value/weight for product {pname}')
    m = Model()
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(display_areas, products, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(sum((values[pname] * x[did, pname] for did in display_areas for pname in products)), GRB.MAXIMIZE)
    for did in display_areas:
        m.addConstr(sum((weights[pname] * x[did, pname] for pname in products)) <= capacities[did], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal', m.ObjVal)
        for did in display_areas:
            for pname in products:
                var = x[did, pname]
                print(var.VarName, var.X)
    else:
        print('Status', m.Status)
    return m
m = solve_boat_display(LEGACY_RECORDS)