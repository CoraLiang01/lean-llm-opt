LEGACY_OBSERVATION = '{"values": {"Capacity": "4120"}}\n\n{"values": {"ProductName": "NSAIDs", "Value": "585", "Weight": "50"}}\n\n{"values": {"ProductName": "Antirheumatic Drugs", "Value": "557", "Weight": "329"}}\n\n{"values": {"ProductName": "Acetic Acid Derivatives", "Value": "963", "Weight": "410"}}\n\n{"values": {"ProductName": "Antibiotics", "Value": "301", "Weight": "452"}}\n\n{"values": {"ProductName": "Antiviral Drugs", "Value": "425", "Weight": "350"}}\n\n{"values": {"ProductName": "Antifungal Agents", "Value": "260", "Weight": "159"}}\n\n{"values": {"ProductName": "Antidepressants", "Value": "848", "Weight": "353"}}\n\n{"values": {"ProductName": "Antipsychotics", "Value": "461", "Weight": "291"}}\n\n{"values": {"ProductName": "Antihistamines", "Value": "840", "Weight": "302"}}\n\n{"values": {"ProductName": "Corticosteroids", "Value": "999", "Weight": "50"}}\n\n{"values": {"ProductName": "Beta Blockers", "Value": "392", "Weight": "250"}}\n\n{"values": {"ProductName": "Calcium Channel Blockers", "Value": "874", "Weight": "178"}}\n\n{"values": {"ProductName": "ACE Inhibitors", "Value": "695", "Weight": "313"}}\n\n{"values": {"ProductName": "Angiotensin II Receptor Blockers", "Value": "405", "Weight": "378"}}\n\n{"values": {"ProductName": "Diuretics", "Value": "320", "Weight": "94"}}\n\n{"values": {"ProductName": "Statins", "Value": "913", "Weight": "97"}}\n\n{"values": {"ProductName": "Insulin", "Value": "754", "Weight": "470"}}\n\n{"values": {"ProductName": "Anticoagulants", "Value": "428", "Weight": "341"}}\n\n{"values": {"ProductName": "Antiepileptic Drugs", "Value": "711", "Weight": "121"}}\n\n{"values": {"ProductName": "Antiemetics", "Value": "998", "Weight": "61"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Capacity': '4120'}}, {'source': '', 'values': {'ProductName': 'NSAIDs', 'Value': '585', 'Weight': '50'}}, {'source': '', 'values': {'ProductName': 'Antirheumatic Drugs', 'Value': '557', 'Weight': '329'}}, {'source': '', 'values': {'ProductName': 'Acetic Acid Derivatives', 'Value': '963', 'Weight': '410'}}, {'source': '', 'values': {'ProductName': 'Antibiotics', 'Value': '301', 'Weight': '452'}}, {'source': '', 'values': {'ProductName': 'Antiviral Drugs', 'Value': '425', 'Weight': '350'}}, {'source': '', 'values': {'ProductName': 'Antifungal Agents', 'Value': '260', 'Weight': '159'}}, {'source': '', 'values': {'ProductName': 'Antidepressants', 'Value': '848', 'Weight': '353'}}, {'source': '', 'values': {'ProductName': 'Antipsychotics', 'Value': '461', 'Weight': '291'}}, {'source': '', 'values': {'ProductName': 'Antihistamines', 'Value': '840', 'Weight': '302'}}, {'source': '', 'values': {'ProductName': 'Corticosteroids', 'Value': '999', 'Weight': '50'}}, {'source': '', 'values': {'ProductName': 'Beta Blockers', 'Value': '392', 'Weight': '250'}}, {'source': '', 'values': {'ProductName': 'Calcium Channel Blockers', 'Value': '874', 'Weight': '178'}}, {'source': '', 'values': {'ProductName': 'ACE Inhibitors', 'Value': '695', 'Weight': '313'}}, {'source': '', 'values': {'ProductName': 'Angiotensin II Receptor Blockers', 'Value': '405', 'Weight': '378'}}, {'source': '', 'values': {'ProductName': 'Diuretics', 'Value': '320', 'Weight': '94'}}, {'source': '', 'values': {'ProductName': 'Statins', 'Value': '913', 'Weight': '97'}}, {'source': '', 'values': {'ProductName': 'Insulin', 'Value': '754', 'Weight': '470'}}, {'source': '', 'values': {'ProductName': 'Anticoagulants', 'Value': '428', 'Weight': '341'}}, {'source': '', 'values': {'ProductName': 'Antiepileptic Drugs', 'Value': '711', 'Weight': '121'}}, {'source': '', 'values': {'ProductName': 'Antiemetics', 'Value': '998', 'Weight': '61'}}]
import gurobipy as gp
from gurobipy import GRB
capacity = None
products = []
values = {}
weights = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    if 'Capacity' in vals and vals['Capacity']:
        if capacity is not None:
            raise ValueError('Multiple capacities found in LEGACY_RECORDS')
        capacity = int(vals['Capacity'])
    elif 'ProductName' in vals and vals['ProductName']:
        pname = vals['ProductName']
        products.append(pname)
        if 'Value' not in vals or 'Weight' not in vals:
            raise ValueError(f'Missing Value or Weight for product {pname}')
        values[pname] = int(vals['Value'])
        weights[pname] = int(vals['Weight'])
if capacity is None:
    raise ValueError('No capacity found in LEGACY_RECORDS')
if len(products) == 0:
    raise ValueError('No products found in LEGACY_RECORDS')
if set(products) != set(values.keys()) or set(products) != set(weights.keys()):
    raise ValueError('Mismatch in product identifiers and coefficients')
m = gp.Model('PharmacyDrugOrder')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[p] * x[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[p] * x[p] for p in products)) <= capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')