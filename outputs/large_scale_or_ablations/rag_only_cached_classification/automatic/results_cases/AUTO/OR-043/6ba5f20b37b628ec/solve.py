LEGACY_OBSERVATION = '{"values": {"Capacity": "520"}}\n{"values": {"ProductName": "NSAIDs", "Value": "250", "Weight": "913"}}\n{"values": {"ProductName": "Antirheumatic Drugs", "Value": "178", "Weight": "754"}}\n{"values": {"ProductName": "Acetic Acid Derivatives", "Value": "313", "Weight": "428"}}\n{"values": {"ProductName": "Antibiotics", "Value": "301", "Weight": "711"}}\n{"values": {"ProductName": "Antiviral Drugs", "Value": "425", "Weight": "350"}}\n{"values": {"ProductName": "Antifungal Agents", "Value": "260", "Weight": "159"}}\n{"values": {"ProductName": "Antidepressants", "Value": "848", "Weight": "353"}}\n{"values": {"ProductName": "Antipsychotics", "Value": "934", "Weight": "291"}}\n{"values": {"ProductName": "Antihistamines", "Value": "114", "Weight": "302"}}\n{"values": {"ProductName": "Corticosteroids", "Value": "1357", "Weight": "50"}}\n{"values": {"ProductName": "Beta Blockers", "Value": "156", "Weight": "250"}}\n{"values": {"ProductName": "Calcium Channel Blockers", "Value": "1780", "Weight": "178"}}\n{"values": {"ProductName": "ACE Inhibitors", "Value": "695", "Weight": "313"}}\n{"values": {"ProductName": "Angiotensin II Receptor Blockers", "Value": "405", "Weight": "378"}}\n{"values": {"ProductName": "Diuretics", "Value": "320", "Weight": "94"}}\n{"values": {"ProductName": "Statins", "Value": "320", "Weight": "97"}}\n{"values": {"ProductName": "Insulin", "Value": "1357", "Weight": "470"}}\n{"values": {"ProductName": "Anticoagulants", "Value": "1357", "Weight": "341"}}\n{"values": {"ProductName": "Antiepileptic Drugs", "Value": "405", "Weight": "121"}}\n{"values": {"ProductName": "Antiemetics", "Value": "998", "Weight": "61"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Capacity': '520'}}, {'source': '', 'values': {'ProductName': 'NSAIDs', 'Value': '250', 'Weight': '913'}}, {'source': '', 'values': {'ProductName': 'Antirheumatic Drugs', 'Value': '178', 'Weight': '754'}}, {'source': '', 'values': {'ProductName': 'Acetic Acid Derivatives', 'Value': '313', 'Weight': '428'}}, {'source': '', 'values': {'ProductName': 'Antibiotics', 'Value': '301', 'Weight': '711'}}, {'source': '', 'values': {'ProductName': 'Antiviral Drugs', 'Value': '425', 'Weight': '350'}}, {'source': '', 'values': {'ProductName': 'Antifungal Agents', 'Value': '260', 'Weight': '159'}}, {'source': '', 'values': {'ProductName': 'Antidepressants', 'Value': '848', 'Weight': '353'}}, {'source': '', 'values': {'ProductName': 'Antipsychotics', 'Value': '934', 'Weight': '291'}}, {'source': '', 'values': {'ProductName': 'Antihistamines', 'Value': '114', 'Weight': '302'}}, {'source': '', 'values': {'ProductName': 'Corticosteroids', 'Value': '1357', 'Weight': '50'}}, {'source': '', 'values': {'ProductName': 'Beta Blockers', 'Value': '156', 'Weight': '250'}}, {'source': '', 'values': {'ProductName': 'Calcium Channel Blockers', 'Value': '1780', 'Weight': '178'}}, {'source': '', 'values': {'ProductName': 'ACE Inhibitors', 'Value': '695', 'Weight': '313'}}, {'source': '', 'values': {'ProductName': 'Angiotensin II Receptor Blockers', 'Value': '405', 'Weight': '378'}}, {'source': '', 'values': {'ProductName': 'Diuretics', 'Value': '320', 'Weight': '94'}}, {'source': '', 'values': {'ProductName': 'Statins', 'Value': '320', 'Weight': '97'}}, {'source': '', 'values': {'ProductName': 'Insulin', 'Value': '1357', 'Weight': '470'}}, {'source': '', 'values': {'ProductName': 'Anticoagulants', 'Value': '1357', 'Weight': '341'}}, {'source': '', 'values': {'ProductName': 'Antiepileptic Drugs', 'Value': '405', 'Weight': '121'}}, {'source': '', 'values': {'ProductName': 'Antiemetics', 'Value': '998', 'Weight': '61'}}]
from gurobipy import Model, GRB

def solve_pharmacy_inventory():
    products = []
    capacity = None
    for rec in LEGACY_RECORDS:
        vals = rec['values']
        if 'Capacity' in vals and vals['Capacity']:
            if capacity is not None:
                raise ValueError('Multiple capacities found in LEGACY_RECORDS')
            capacity = int(vals['Capacity'])
        elif 'ProductName' in vals and vals['ProductName']:
            try:
                pname = vals['ProductName']
                value = int(vals['Value'])
                weight = int(vals['Weight'])
            except Exception as e:
                raise ValueError(f'Invalid product record: {vals}') from e
            products.append({'ProductName': pname, 'Value': value, 'Weight': weight})
    if capacity is None:
        raise ValueError('No capacity found in LEGACY_RECORDS')
    if len(products) != 20:
        raise ValueError(f'Expected 20 products, found {len(products)}')
    for p in products:
        if not all((k in p for k in ('ProductName', 'Value', 'Weight'))):
            raise ValueError(f'Missing fields in product: {p}')
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(range(len(products)), vtype=GRB.INTEGER, lb=0, name='')
    obj = sum((products[i]['Value'] * x[i] for i in range(len(products))))
    m.setObjective(obj, GRB.MAXIMIZE)
    cap_expr = sum((products[i]['Weight'] * x[i] for i in range(len(products))))
    m.addConstr(cap_expr <= capacity, name='cap')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for i in range(len(products)):
            v = x[i]
            print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
m = solve_pharmacy_inventory()