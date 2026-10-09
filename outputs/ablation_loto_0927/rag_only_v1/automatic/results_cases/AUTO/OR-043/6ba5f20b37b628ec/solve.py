LEGACY_OBSERVATION = '{"values": {"ProductName": "NSAIDs", "Value": "250", "Weight": "913"}}\n{"values": {"ProductName": "Antirheumatic Drugs", "Value": "178", "Weight": "754"}}\n{"values": {"ProductName": "Acetic Acid Derivatives", "Value": "313", "Weight": "428"}}\n{"values": {"ProductName": "Antibiotics", "Value": "301", "Weight": "711"}}\n{"values": {"ProductName": "Antiviral Drugs", "Value": "425", "Weight": "350"}}\n{"values": {"ProductName": "Antifungal Agents", "Value": "260", "Weight": "159"}}\n{"values": {"ProductName": "Antidepressants", "Value": "848", "Weight": "353"}}\n{"values": {"ProductName": "Antipsychotics", "Value": "934", "Weight": "291"}}\n{"values": {"ProductName": "Antihistamines", "Value": "114", "Weight": "302"}}\n{"values": {"ProductName": "Corticosteroids", "Value": "1357", "Weight": "50"}}\n{"values": {"ProductName": "Beta Blockers", "Value": "156", "Weight": "250"}}\n{"values": {"ProductName": "Calcium Channel Blockers", "Value": "1780", "Weight": "178"}}\n{"values": {"ProductName": "ACE Inhibitors", "Value": "695", "Weight": "313"}}\n{"values": {"ProductName": "Angiotensin II Receptor Blockers", "Value": "405", "Weight": "378"}}\n{"values": {"ProductName": "Diuretics", "Value": "320", "Weight": "94"}}\n{"values": {"ProductName": "Statins", "Value": "320", "Weight": "97"}}\n{"values": {"ProductName": "Insulin", "Value": "1357", "Weight": "470"}}\n{"values": {"ProductName": "Anticoagulants", "Value": "1357", "Weight": "341"}}\n{"values": {"ProductName": "Antiepileptic Drugs", "Value": "405", "Weight": "121"}}\n{"values": {"ProductName": "Antiemetics", "Value": "998", "Weight": "61"}}\n{"values": {"Capacity": "520"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'ProductName': 'NSAIDs', 'Value': '250', 'Weight': '913'}}, {'source': '', 'values': {'ProductName': 'Antirheumatic Drugs', 'Value': '178', 'Weight': '754'}}, {'source': '', 'values': {'ProductName': 'Acetic Acid Derivatives', 'Value': '313', 'Weight': '428'}}, {'source': '', 'values': {'ProductName': 'Antibiotics', 'Value': '301', 'Weight': '711'}}, {'source': '', 'values': {'ProductName': 'Antiviral Drugs', 'Value': '425', 'Weight': '350'}}, {'source': '', 'values': {'ProductName': 'Antifungal Agents', 'Value': '260', 'Weight': '159'}}, {'source': '', 'values': {'ProductName': 'Antidepressants', 'Value': '848', 'Weight': '353'}}, {'source': '', 'values': {'ProductName': 'Antipsychotics', 'Value': '934', 'Weight': '291'}}, {'source': '', 'values': {'ProductName': 'Antihistamines', 'Value': '114', 'Weight': '302'}}, {'source': '', 'values': {'ProductName': 'Corticosteroids', 'Value': '1357', 'Weight': '50'}}, {'source': '', 'values': {'ProductName': 'Beta Blockers', 'Value': '156', 'Weight': '250'}}, {'source': '', 'values': {'ProductName': 'Calcium Channel Blockers', 'Value': '1780', 'Weight': '178'}}, {'source': '', 'values': {'ProductName': 'ACE Inhibitors', 'Value': '695', 'Weight': '313'}}, {'source': '', 'values': {'ProductName': 'Angiotensin II Receptor Blockers', 'Value': '405', 'Weight': '378'}}, {'source': '', 'values': {'ProductName': 'Diuretics', 'Value': '320', 'Weight': '94'}}, {'source': '', 'values': {'ProductName': 'Statins', 'Value': '320', 'Weight': '97'}}, {'source': '', 'values': {'ProductName': 'Insulin', 'Value': '1357', 'Weight': '470'}}, {'source': '', 'values': {'ProductName': 'Anticoagulants', 'Value': '1357', 'Weight': '341'}}, {'source': '', 'values': {'ProductName': 'Antiepileptic Drugs', 'Value': '405', 'Weight': '121'}}, {'source': '', 'values': {'ProductName': 'Antiemetics', 'Value': '998', 'Weight': '61'}}, {'source': '', 'values': {'Capacity': '520'}}]
import gurobipy as gp
from gurobipy import GRB

def solve_pharmacy_inventory():
    products = []
    capacity = None
    for rec in LEGACY_RECORDS:
        if rec['source'] == '' and 'ProductName' in rec['values']:
            products.append({'ProductName': rec['values']['ProductName'], 'Value': int(rec['values']['Value']), 'Weight': int(rec['values']['Weight'])})
        elif rec['source'] == '' and 'Capacity' in rec['values']:
            capacity = int(rec['values']['Capacity'])
    if len(products) == 0 or capacity is None:
        raise ValueError('Missing product or capacity data in LEGACY_RECORDS.')
    for p in products:
        if not all((k in p for k in ('ProductName', 'Value', 'Weight'))):
            raise ValueError(f'Missing fields in product: {p}')
    product_keys = [p['ProductName'] for p in products]
    value = {p['ProductName']: p['Value'] for p in products}
    weight = {p['ProductName']: p['Weight'] for p in products}
    m = gp.Model()
    x = m.addVars(product_keys, vtype=GRB.INTEGER, lb=0, obj=0, name='')
    m.setObjective(gp.quicksum((value[k] * x[k] for k in product_keys)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight[k] * x[k] for k in product_keys)) <= capacity, name='')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for k in product_keys:
            print(f'{x[k].VarName} {x[k].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_pharmacy_inventory()