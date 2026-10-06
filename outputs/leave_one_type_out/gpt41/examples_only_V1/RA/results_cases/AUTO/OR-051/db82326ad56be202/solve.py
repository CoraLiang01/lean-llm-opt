LEGACY_OBSERVATION = 'capacity.csv\nCabinetID,Capacity\n1,400\n2,600\n3,500\n4,700\n5,450\n6,650\n7,550\n8,750\n9,480\n10,520\n\nproducts.csv\nProductName,Value,Weight\nEspresso Beans,100,1.0\nColombian Roast,150,1.5\nArabica Blend,80,1.2\nFrench Roast,120,1.3\nItalian Roast,130,1.4\nHouse Blend,110,1.1\nSumatra Coffee,160,1.8\nMocha Java,90,1.2\nHazelnut Flavor,95,1.0\nCaramel Blend,105,1.3\nVanilla Flavor,85,1.2\nCappuccino Mix,140,1.5\nPumpkin Spice,75,1.1\nDecaf Roast,60,1.0\nOrganic Roast,170,1.6\nCold Brew,115,1.4\nPeruvian Blend,155,1.7\nKenyan AA,125,1.3'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'CabinetID': '1', 'Capacity': '400'}}, {'source': 'capacity.csv', 'values': {'CabinetID': '2', 'Capacity': '600'}}, {'source': 'capacity.csv', 'values': {'CabinetID': '3', 'Capacity': '500'}}, {'source': 'capacity.csv', 'values': {'CabinetID': '4', 'Capacity': '700'}}, {'source': 'capacity.csv', 'values': {'CabinetID': '5', 'Capacity': '450'}}, {'source': 'capacity.csv', 'values': {'CabinetID': '6', 'Capacity': '650'}}, {'source': 'capacity.csv', 'values': {'CabinetID': '7', 'Capacity': '550'}}, {'source': 'capacity.csv', 'values': {'CabinetID': '8', 'Capacity': '750'}}, {'source': 'capacity.csv', 'values': {'CabinetID': '9', 'Capacity': '480'}}, {'source': 'capacity.csv', 'values': {'CabinetID': '10', 'Capacity': '520'}}, {'source': 'products.csv', 'values': {'ProductName': 'Espresso Beans', 'Value': '100', 'Weight': '1.0'}}, {'source': 'products.csv', 'values': {'ProductName': 'Colombian Roast', 'Value': '150', 'Weight': '1.5'}}, {'source': 'products.csv', 'values': {'ProductName': 'Arabica Blend', 'Value': '80', 'Weight': '1.2'}}, {'source': 'products.csv', 'values': {'ProductName': 'French Roast', 'Value': '120', 'Weight': '1.3'}}, {'source': 'products.csv', 'values': {'ProductName': 'Italian Roast', 'Value': '130', 'Weight': '1.4'}}, {'source': 'products.csv', 'values': {'ProductName': 'House Blend', 'Value': '110', 'Weight': '1.1'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sumatra Coffee', 'Value': '160', 'Weight': '1.8'}}, {'source': 'products.csv', 'values': {'ProductName': 'Mocha Java', 'Value': '90', 'Weight': '1.2'}}, {'source': 'products.csv', 'values': {'ProductName': 'Hazelnut Flavor', 'Value': '95', 'Weight': '1.0'}}, {'source': 'products.csv', 'values': {'ProductName': 'Caramel Blend', 'Value': '105', 'Weight': '1.3'}}, {'source': 'products.csv', 'values': {'ProductName': 'Vanilla Flavor', 'Value': '85', 'Weight': '1.2'}}, {'source': 'products.csv', 'values': {'ProductName': 'Cappuccino Mix', 'Value': '140', 'Weight': '1.5'}}, {'source': 'products.csv', 'values': {'ProductName': 'Pumpkin Spice', 'Value': '75', 'Weight': '1.1'}}, {'source': 'products.csv', 'values': {'ProductName': 'Decaf Roast', 'Value': '60', 'Weight': '1.0'}}, {'source': 'products.csv', 'values': {'ProductName': 'Organic Roast', 'Value': '170', 'Weight': '1.6'}}, {'source': 'products.csv', 'values': {'ProductName': 'Cold Brew', 'Value': '115', 'Weight': '1.4'}}, {'source': 'products.csv', 'values': {'ProductName': 'Peruvian Blend', 'Value': '155', 'Weight': '1.7'}}, {'source': 'products.csv', 'values': {'ProductName': 'Kenyan AA', 'Value': '125', 'Weight': '1.3'}}]
from gurobipy import Model, GRB
cabinets = []
capacities = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == 'capacity.csv':
        cid = int(rec['values']['CabinetID'])
        cap = float(rec['values']['Capacity'])
        cabinets.append(cid)
        capacities[cid] = cap
products = []
values = {}
weights = {}
product_names = {}
for idx, rec in enumerate(LEGACY_RECORDS):
    if rec['source'] == 'products.csv':
        j = idx - sum((1 for r in LEGACY_RECORDS[:idx] if r['source'] == 'capacity.csv')) + 1
        pname = rec['values']['ProductName']
        val = float(rec['values']['Value'])
        wgt = float(rec['values']['Weight'])
        products.append(j)
        values[j] = val
        weights[j] = wgt
        product_names[j] = pname
if len(cabinets) == 0 or len(products) == 0:
    raise RuntimeError('Missing cabinets or products data.')
for cid in cabinets:
    if cid not in capacities:
        raise RuntimeError(f'Missing capacity for cabinet {cid}.')
for j in products:
    if j not in values or j not in weights:
        raise RuntimeError(f'Missing value or weight for product {j}.')
m = Model()
m.setParam('MIPGap', 0.0001)
x = m.addVars(cabinets, products, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(sum((values[j] * x[i, j] for i in cabinets for j in products)), GRB.MAXIMIZE)
for i in cabinets:
    m.addConstr(sum((weights[j] * x[i, j] for j in products)) <= capacities[i], name='cap_%d' % i)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('ObjVal:', m.ObjVal)
    for i in cabinets:
        for j in products:
            var = x[i, j]
            print(f'{var.VarName} {var.X}')
else:
    print('Solver status:', m.Status)