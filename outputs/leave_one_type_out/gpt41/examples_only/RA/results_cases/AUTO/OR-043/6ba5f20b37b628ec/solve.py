LEGACY_OBSERVATION = 'capacity.csv\nCapacity\n520\n\nproducts.csv\nProductName,Value,Weight\nNSAIDs,250,913\nAntirheumatic Drugs,178,754\nAcetic Acid Derivatives,313,428\nAntibiotics,301,711\nAntiviral Drugs,425,350\nAntifungal Agents,260,159\nAntidepressants,848,353\nAntipsychotics,934,291\nAntihistamines,114,302\nCorticosteroids,1357,50\nBeta Blockers,156,250\nCalcium Channel Blockers,1780,178\nACE Inhibitors,695,313\nAngiotensin II Receptor Blockers,405,378\nDiuretics,320,94\nStatins,320,97\nInsulin,1357,470\nAnticoagulants,1357,341\nAntiepileptic Drugs,405,121\nAntiemetics,998,61'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'Capacity': '520'}}, {'source': 'products.csv', 'values': {'ProductName': 'NSAIDs', 'Value': '250', 'Weight': '913'}}, {'source': 'products.csv', 'values': {'ProductName': 'Antirheumatic Drugs', 'Value': '178', 'Weight': '754'}}, {'source': 'products.csv', 'values': {'ProductName': 'Acetic Acid Derivatives', 'Value': '313', 'Weight': '428'}}, {'source': 'products.csv', 'values': {'ProductName': 'Antibiotics', 'Value': '301', 'Weight': '711'}}, {'source': 'products.csv', 'values': {'ProductName': 'Antiviral Drugs', 'Value': '425', 'Weight': '350'}}, {'source': 'products.csv', 'values': {'ProductName': 'Antifungal Agents', 'Value': '260', 'Weight': '159'}}, {'source': 'products.csv', 'values': {'ProductName': 'Antidepressants', 'Value': '848', 'Weight': '353'}}, {'source': 'products.csv', 'values': {'ProductName': 'Antipsychotics', 'Value': '934', 'Weight': '291'}}, {'source': 'products.csv', 'values': {'ProductName': 'Antihistamines', 'Value': '114', 'Weight': '302'}}, {'source': 'products.csv', 'values': {'ProductName': 'Corticosteroids', 'Value': '1357', 'Weight': '50'}}, {'source': 'products.csv', 'values': {'ProductName': 'Beta Blockers', 'Value': '156', 'Weight': '250'}}, {'source': 'products.csv', 'values': {'ProductName': 'Calcium Channel Blockers', 'Value': '1780', 'Weight': '178'}}, {'source': 'products.csv', 'values': {'ProductName': 'ACE Inhibitors', 'Value': '695', 'Weight': '313'}}, {'source': 'products.csv', 'values': {'ProductName': 'Angiotensin II Receptor Blockers', 'Value': '405', 'Weight': '378'}}, {'source': 'products.csv', 'values': {'ProductName': 'Diuretics', 'Value': '320', 'Weight': '94'}}, {'source': 'products.csv', 'values': {'ProductName': 'Statins', 'Value': '320', 'Weight': '97'}}, {'source': 'products.csv', 'values': {'ProductName': 'Insulin', 'Value': '1357', 'Weight': '470'}}, {'source': 'products.csv', 'values': {'ProductName': 'Anticoagulants', 'Value': '1357', 'Weight': '341'}}, {'source': 'products.csv', 'values': {'ProductName': 'Antiepileptic Drugs', 'Value': '405', 'Weight': '121'}}, {'source': 'products.csv', 'values': {'ProductName': 'Antiemetics', 'Value': '998', 'Weight': '61'}}]
from gurobipy import Model, GRB

def solve_pharmacy_inventory():
    global LEGACY_RECORDS
    capacity_records = [r for r in LEGACY_RECORDS if r['source'] == 'capacity.csv']
    if not capacity_records or 'Capacity' not in capacity_records[0]['values']:
        raise ValueError('Missing capacity in LEGACY_RECORDS')
    try:
        capacity = int(capacity_records[0]['values']['Capacity'])
    except Exception:
        raise ValueError('Capacity value is not an integer')
    product_records = [r for r in LEGACY_RECORDS if r['source'] == 'products.csv']
    if len(product_records) != 20:
        raise ValueError('Expected 20 products in LEGACY_RECORDS')
    products = []
    for rec in product_records:
        vals = rec['values']
        if not all((k in vals for k in ('ProductName', 'Value', 'Weight'))):
            raise ValueError('Missing product fields in LEGACY_RECORDS')
        try:
            products.append({'ProductName': vals['ProductName'], 'Value': int(vals['Value']), 'Weight': int(vals['Weight'])})
        except Exception:
            raise ValueError('Non-integer Value or Weight in LEGACY_RECORDS')
    if len(products) != 20:
        raise ValueError('Product count mismatch')
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars([p['ProductName'] for p in products], vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(sum((p['Value'] * x[p['ProductName']] for p in products)), GRB.MAXIMIZE)
    m.addConstr(sum((p['Weight'] * x[p['ProductName']] for p in products)) <= capacity, name='cap')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for p in products:
            v = x[p['ProductName']]
            print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_pharmacy_inventory()