LEGACY_OBSERVATION = 'capacity.csv\n{"values": {"CabinetID": "1", "Capacity": "400"}}\n{"values": {"CabinetID": "2", "Capacity": "600"}}\n{"values": {"CabinetID": "3", "Capacity": "500"}}\n{"values": {"CabinetID": "4", "Capacity": "700"}}\n{"values": {"CabinetID": "5", "Capacity": "450"}}\n{"values": {"CabinetID": "6", "Capacity": "650"}}\n{"values": {"CabinetID": "7", "Capacity": "550"}}\n{"values": {"CabinetID": "8", "Capacity": "750"}}\n{"values": {"CabinetID": "9", "Capacity": "480"}}\n{"values": {"CabinetID": "10", "Capacity": "520"}}\n\nproducts.csv\n{"values": {"ProductName": "Espresso Beans", "Value": "100", "Weight": "1.0"}}\n{"values": {"ProductName": "Colombian Roast", "Value": "150", "Weight": "1.5"}}\n{"values": {"ProductName": "Arabica Blend", "Value": "80", "Weight": "1.2"}}\n{"values": {"ProductName": "French Roast", "Value": "120", "Weight": "1.3"}}\n{"values": {"ProductName": "Italian Roast", "Value": "130", "Weight": "1.4"}}\n{"values": {"ProductName": "House Blend", "Value": "110", "Weight": "1.1"}}\n{"values": {"ProductName": "Sumatra Coffee", "Value": "160", "Weight": "1.8"}}\n{"values": {"ProductName": "Mocha Java", "Value": "90", "Weight": "1.2"}}\n{"values": {"ProductName": "Hazelnut Flavor", "Value": "95", "Weight": "1.0"}}\n{"values": {"ProductName": "Caramel Blend", "Value": "105", "Weight": "1.3"}}\n{"values": {"ProductName": "Vanilla Flavor", "Value": "85", "Weight": "1.2"}}\n{"values": {"ProductName": "Cappuccino Mix", "Value": "140", "Weight": "1.5"}}\n{"values": {"ProductName": "Pumpkin Spice", "Value": "75", "Weight": "1.1"}}\n{"values": {"ProductName": "Decaf Roast", "Value": "60", "Weight": "1.0"}}\n{"values": {"ProductName": "Organic Roast", "Value": "170", "Weight": "1.6"}}\n{"values": {"ProductName": "Cold Brew", "Value": "115", "Weight": "1.4"}}\n{"values": {"ProductName": "Peruvian Blend", "Value": "155", "Weight": "1.7"}}\n{"values": {"ProductName": "Kenyan AA", "Value": "125", "Weight": "1.3"}}'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'CabinetID': '1', 'Capacity': '400'}}, {'source': 'capacity.csv', 'values': {'CabinetID': '2', 'Capacity': '600'}}, {'source': 'capacity.csv', 'values': {'CabinetID': '3', 'Capacity': '500'}}, {'source': 'capacity.csv', 'values': {'CabinetID': '4', 'Capacity': '700'}}, {'source': 'capacity.csv', 'values': {'CabinetID': '5', 'Capacity': '450'}}, {'source': 'capacity.csv', 'values': {'CabinetID': '6', 'Capacity': '650'}}, {'source': 'capacity.csv', 'values': {'CabinetID': '7', 'Capacity': '550'}}, {'source': 'capacity.csv', 'values': {'CabinetID': '8', 'Capacity': '750'}}, {'source': 'capacity.csv', 'values': {'CabinetID': '9', 'Capacity': '480'}}, {'source': 'capacity.csv', 'values': {'CabinetID': '10', 'Capacity': '520'}}, {'source': 'products.csv', 'values': {'ProductName': 'Espresso Beans', 'Value': '100', 'Weight': '1.0'}}, {'source': 'products.csv', 'values': {'ProductName': 'Colombian Roast', 'Value': '150', 'Weight': '1.5'}}, {'source': 'products.csv', 'values': {'ProductName': 'Arabica Blend', 'Value': '80', 'Weight': '1.2'}}, {'source': 'products.csv', 'values': {'ProductName': 'French Roast', 'Value': '120', 'Weight': '1.3'}}, {'source': 'products.csv', 'values': {'ProductName': 'Italian Roast', 'Value': '130', 'Weight': '1.4'}}, {'source': 'products.csv', 'values': {'ProductName': 'House Blend', 'Value': '110', 'Weight': '1.1'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sumatra Coffee', 'Value': '160', 'Weight': '1.8'}}, {'source': 'products.csv', 'values': {'ProductName': 'Mocha Java', 'Value': '90', 'Weight': '1.2'}}, {'source': 'products.csv', 'values': {'ProductName': 'Hazelnut Flavor', 'Value': '95', 'Weight': '1.0'}}, {'source': 'products.csv', 'values': {'ProductName': 'Caramel Blend', 'Value': '105', 'Weight': '1.3'}}, {'source': 'products.csv', 'values': {'ProductName': 'Vanilla Flavor', 'Value': '85', 'Weight': '1.2'}}, {'source': 'products.csv', 'values': {'ProductName': 'Cappuccino Mix', 'Value': '140', 'Weight': '1.5'}}, {'source': 'products.csv', 'values': {'ProductName': 'Pumpkin Spice', 'Value': '75', 'Weight': '1.1'}}, {'source': 'products.csv', 'values': {'ProductName': 'Decaf Roast', 'Value': '60', 'Weight': '1.0'}}, {'source': 'products.csv', 'values': {'ProductName': 'Organic Roast', 'Value': '170', 'Weight': '1.6'}}, {'source': 'products.csv', 'values': {'ProductName': 'Cold Brew', 'Value': '115', 'Weight': '1.4'}}, {'source': 'products.csv', 'values': {'ProductName': 'Peruvian Blend', 'Value': '155', 'Weight': '1.7'}}, {'source': 'products.csv', 'values': {'ProductName': 'Kenyan AA', 'Value': '125', 'Weight': '1.3'}}]
from gurobipy import Model, GRB

def solve_coffee_allocation():
    global LEGACY_RECORDS
    cabinets = []
    capacities = []
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'capacity.csv':
            cabinets.append(rec['values']['CabinetID'])
            capacities.append(float(rec['values']['Capacity']))
    if len(cabinets) != len(capacities):
        raise ValueError('Mismatch in cabinets and capacities length')
    products = []
    values = []
    weights = []
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'products.csv':
            products.append(rec['values']['ProductName'])
            values.append(float(rec['values']['Value']))
            weights.append(float(rec['values']['Weight']))
    if len(products) != len(values) or len(products) != len(weights):
        raise ValueError('Mismatch in products, values, or weights length')
    n_cab = len(cabinets)
    n_prod = len(products)
    if n_cab == 0 or n_prod == 0:
        raise ValueError('No cabinets or products found in LEGACY_RECORDS')
    cab_ids = cabinets
    prod_names = products
    cab_caps = dict(zip(cab_ids, capacities))
    prod_vals = dict(zip(prod_names, values))
    prod_wts = dict(zip(prod_names, weights))
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(cab_ids, prod_names, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(sum((prod_vals[j] * x[i, j] for i in cab_ids for j in prod_names)), GRB.MAXIMIZE)
    for (idx, i) in enumerate(cab_ids):
        m.addConstr(sum((prod_wts[j] * x[i, j] for j in prod_names)) <= cab_caps[i], name='cap_%s' % i)
    for i in cab_ids:
        if i not in cab_caps:
            raise ValueError(f'Missing capacity for cabinet {i}')
    for j in prod_names:
        if j not in prod_vals or j not in prod_wts:
            raise ValueError(f'Missing value or weight for product {j}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal:', m.ObjVal)
        for i in cab_ids:
            for j in prod_names:
                v = x[i, j]
                print(f'{v.VarName} {v.X}')
    else:
        print('Solver status:', m.Status)
m = solve_coffee_allocation()