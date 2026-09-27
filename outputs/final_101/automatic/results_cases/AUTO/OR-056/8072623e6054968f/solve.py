LEGACY_OBSERVATION = 'capacity.csv\nDisplayID,Capacity\n1,457\n2,604\n3,751\n4,468\n5,343\n6,408\n7,741\n8,914\n9,682\n10,409\n11,342\n12,903\n13,680\n14,886\n\nproducts.csv\nProductName,Value,Weight\nSpeedboat,29664,18\nFishing Boat,31778,36\nCatamaran,73501,25\nYacht,78255,16\nSailboat,93606,97\nKayak,46983,35\nCanoe,95026,32\nHouseboat,57685,100\nPontoon,60323,43\nJet Ski,91224,15\nRowboat,44003,95\nHovercraft,75998,57\nCabin Cruiser,84525,13\nWakeboard Boat,66207,44\nDinghy,65002,64\nTrawler,33132,88\nPaddle Boat,69239,42\nSubmarine,66948,46\nRIB,88240,24\nSkiff,48858,93'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'DisplayID': '1', 'Capacity': '457'}}, {'source': 'capacity.csv', 'values': {'DisplayID': '2', 'Capacity': '604'}}, {'source': 'capacity.csv', 'values': {'DisplayID': '3', 'Capacity': '751'}}, {'source': 'capacity.csv', 'values': {'DisplayID': '4', 'Capacity': '468'}}, {'source': 'capacity.csv', 'values': {'DisplayID': '5', 'Capacity': '343'}}, {'source': 'capacity.csv', 'values': {'DisplayID': '6', 'Capacity': '408'}}, {'source': 'capacity.csv', 'values': {'DisplayID': '7', 'Capacity': '741'}}, {'source': 'capacity.csv', 'values': {'DisplayID': '8', 'Capacity': '914'}}, {'source': 'capacity.csv', 'values': {'DisplayID': '9', 'Capacity': '682'}}, {'source': 'capacity.csv', 'values': {'DisplayID': '10', 'Capacity': '409'}}, {'source': 'capacity.csv', 'values': {'DisplayID': '11', 'Capacity': '342'}}, {'source': 'capacity.csv', 'values': {'DisplayID': '12', 'Capacity': '903'}}, {'source': 'capacity.csv', 'values': {'DisplayID': '13', 'Capacity': '680'}}, {'source': 'capacity.csv', 'values': {'DisplayID': '14', 'Capacity': '886'}}, {'source': 'products.csv', 'values': {'ProductName': 'Speedboat', 'Value': '29664', 'Weight': '18'}}, {'source': 'products.csv', 'values': {'ProductName': 'Fishing Boat', 'Value': '31778', 'Weight': '36'}}, {'source': 'products.csv', 'values': {'ProductName': 'Catamaran', 'Value': '73501', 'Weight': '25'}}, {'source': 'products.csv', 'values': {'ProductName': 'Yacht', 'Value': '78255', 'Weight': '16'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sailboat', 'Value': '93606', 'Weight': '97'}}, {'source': 'products.csv', 'values': {'ProductName': 'Kayak', 'Value': '46983', 'Weight': '35'}}, {'source': 'products.csv', 'values': {'ProductName': 'Canoe', 'Value': '95026', 'Weight': '32'}}, {'source': 'products.csv', 'values': {'ProductName': 'Houseboat', 'Value': '57685', 'Weight': '100'}}, {'source': 'products.csv', 'values': {'ProductName': 'Pontoon', 'Value': '60323', 'Weight': '43'}}, {'source': 'products.csv', 'values': {'ProductName': 'Jet Ski', 'Value': '91224', 'Weight': '15'}}, {'source': 'products.csv', 'values': {'ProductName': 'Rowboat', 'Value': '44003', 'Weight': '95'}}, {'source': 'products.csv', 'values': {'ProductName': 'Hovercraft', 'Value': '75998', 'Weight': '57'}}, {'source': 'products.csv', 'values': {'ProductName': 'Cabin Cruiser', 'Value': '84525', 'Weight': '13'}}, {'source': 'products.csv', 'values': {'ProductName': 'Wakeboard Boat', 'Value': '66207', 'Weight': '44'}}, {'source': 'products.csv', 'values': {'ProductName': 'Dinghy', 'Value': '65002', 'Weight': '64'}}, {'source': 'products.csv', 'values': {'ProductName': 'Trawler', 'Value': '33132', 'Weight': '88'}}, {'source': 'products.csv', 'values': {'ProductName': 'Paddle Boat', 'Value': '69239', 'Weight': '42'}}, {'source': 'products.csv', 'values': {'ProductName': 'Submarine', 'Value': '66948', 'Weight': '46'}}, {'source': 'products.csv', 'values': {'ProductName': 'RIB', 'Value': '88240', 'Weight': '24'}}, {'source': 'products.csv', 'values': {'ProductName': 'Skiff', 'Value': '48858', 'Weight': '93'}}]
import gurobipy as gp
from gurobipy import GRB
capacity_records = [r for r in LEGACY_RECORDS if r['source'] == 'capacity.csv']
product_records = [r for r in LEGACY_RECORDS if r['source'] == 'products.csv']
display_ids = [rec['values']['DisplayID'] for rec in capacity_records]
capacities = {rec['values']['DisplayID']: int(rec['values']['Capacity']) for rec in capacity_records}
product_names = [rec['values']['ProductName'] for rec in product_records]
values = {rec['values']['ProductName']: int(rec['values']['Value']) for rec in product_records}
weights = {rec['values']['ProductName']: int(rec['values']['Weight']) for rec in product_records}
if len(display_ids) != 14 or len(product_names) != 20:
    raise ValueError('Missing display area or product data.')
for did in display_ids:
    if did not in capacities:
        raise ValueError(f'Missing capacity for display area {did}')
for pname in product_names:
    if pname not in values or pname not in weights:
        raise ValueError(f'Missing value or weight for product {pname}')
m = gp.Model('BoatDisplayAssignment')
x = m.addVars(display_ids, product_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[pname] * x[did, pname] for did in display_ids for pname in product_names)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[pname] * x[did, pname] for pname in product_names)) <= capacities[did] for did in display_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')