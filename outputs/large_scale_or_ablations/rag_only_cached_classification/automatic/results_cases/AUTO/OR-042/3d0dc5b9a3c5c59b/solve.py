LEGACY_OBSERVATION = '{"values": {"Capacity": "4120"}}\n{"values": {"ProductName": "NSAIDs", "Value": "585", "Weight": "50"}}\n{"values": {"ProductName": "Antirheumatic Drugs", "Value": "557", "Weight": "329"}}\n{"values": {"ProductName": "Acetic Acid Derivatives", "Value": "963", "Weight": "410"}}\n{"values": {"ProductName": "Antibiotics", "Value": "301", "Weight": "452"}}\n{"values": {"ProductName": "Antiviral Drugs", "Value": "425", "Weight": "350"}}\n{"values": {"ProductName": "Antifungal Agents", "Value": "260", "Weight": "159"}}\n{"values": {"ProductName": "Antidepressants", "Value": "848", "Weight": "353"}}\n{"values": {"ProductName": "Antipsychotics", "Value": "461", "Weight": "291"}}\n{"values": {"ProductName": "Antihistamines", "Value": "840", "Weight": "302"}}\n{"values": {"ProductName": "Corticosteroids", "Value": "999", "Weight": "50"}}\n{"values": {"ProductName": "Beta Blockers", "Value": "392", "Weight": "250"}}\n{"values": {"ProductName": "Calcium Channel Blockers", "Value": "874", "Weight": "178"}}\n{"values": {"ProductName": "ACE Inhibitors", "Value": "695", "Weight": "313"}}\n{"values": {"ProductName": "Angiotensin II Receptor Blockers", "Value": "405", "Weight": "378"}}\n{"values": {"ProductName": "Diuretics", "Value": "320", "Weight": "94"}}\n{"values": {"ProductName": "Statins", "Value": "913", "Weight": "97"}}\n{"values": {"ProductName": "Insulin", "Value": "754", "Weight": "470"}}\n{"values": {"ProductName": "Anticoagulants", "Value": "428", "Weight": "341"}}\n{"values": {"ProductName": "Antiepileptic Drugs", "Value": "711", "Weight": "121"}}\n{"values": {"ProductName": "Antiemetics", "Value": "998", "Weight": "61"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Capacity': '4120'}}, {'source': '', 'values': {'ProductName': 'NSAIDs', 'Value': '585', 'Weight': '50'}}, {'source': '', 'values': {'ProductName': 'Antirheumatic Drugs', 'Value': '557', 'Weight': '329'}}, {'source': '', 'values': {'ProductName': 'Acetic Acid Derivatives', 'Value': '963', 'Weight': '410'}}, {'source': '', 'values': {'ProductName': 'Antibiotics', 'Value': '301', 'Weight': '452'}}, {'source': '', 'values': {'ProductName': 'Antiviral Drugs', 'Value': '425', 'Weight': '350'}}, {'source': '', 'values': {'ProductName': 'Antifungal Agents', 'Value': '260', 'Weight': '159'}}, {'source': '', 'values': {'ProductName': 'Antidepressants', 'Value': '848', 'Weight': '353'}}, {'source': '', 'values': {'ProductName': 'Antipsychotics', 'Value': '461', 'Weight': '291'}}, {'source': '', 'values': {'ProductName': 'Antihistamines', 'Value': '840', 'Weight': '302'}}, {'source': '', 'values': {'ProductName': 'Corticosteroids', 'Value': '999', 'Weight': '50'}}, {'source': '', 'values': {'ProductName': 'Beta Blockers', 'Value': '392', 'Weight': '250'}}, {'source': '', 'values': {'ProductName': 'Calcium Channel Blockers', 'Value': '874', 'Weight': '178'}}, {'source': '', 'values': {'ProductName': 'ACE Inhibitors', 'Value': '695', 'Weight': '313'}}, {'source': '', 'values': {'ProductName': 'Angiotensin II Receptor Blockers', 'Value': '405', 'Weight': '378'}}, {'source': '', 'values': {'ProductName': 'Diuretics', 'Value': '320', 'Weight': '94'}}, {'source': '', 'values': {'ProductName': 'Statins', 'Value': '913', 'Weight': '97'}}, {'source': '', 'values': {'ProductName': 'Insulin', 'Value': '754', 'Weight': '470'}}, {'source': '', 'values': {'ProductName': 'Anticoagulants', 'Value': '428', 'Weight': '341'}}, {'source': '', 'values': {'ProductName': 'Antiepileptic Drugs', 'Value': '711', 'Weight': '121'}}, {'source': '', 'values': {'ProductName': 'Antiemetics', 'Value': '998', 'Weight': '61'}}]
from gurobipy import Model, GRB

def solve_pharmacy_knapsack(LEGACY_RECORDS):
    capacity = None
    products = []
    for rec in LEGACY_RECORDS:
        vals = rec['values']
        if 'Capacity' in vals and vals['Capacity']:
            if capacity is not None:
                raise ValueError('Multiple capacities found')
            capacity = int(vals['Capacity'])
        elif 'ProductName' in vals and vals['ProductName']:
            try:
                products.append({'ProductName': vals['ProductName'], 'Value': int(vals['Value']), 'Weight': int(vals['Weight'])})
            except Exception as e:
                raise ValueError(f'Invalid product record: {vals}') from e
    if capacity is None:
        raise ValueError('No capacity found in LEGACY_RECORDS')
    if len(products) != 20:
        raise ValueError(f'Expected 20 products, got {len(products)}')
    drug_types = [p['ProductName'] for p in products]
    benefit = {p['ProductName']: p['Value'] for p in products}
    weight = {p['ProductName']: p['Weight'] for p in products}
    for k in drug_types:
        if k not in benefit or k not in weight:
            raise ValueError(f'Missing coefficients for {k}')
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(drug_types, vtype=GRB.INTEGER, lb=0, obj=0, name='')
    m.setObjective(sum((benefit[k] * x[k] for k in drug_types)), GRB.MAXIMIZE)
    m.addConstr(sum((weight[k] * x[k] for k in drug_types)) <= capacity, name='cap')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for k in drug_types:
            print(f'{x[k].VarName} {x[k].X}')
    else:
        print(f'Status {m.Status}')
    return m