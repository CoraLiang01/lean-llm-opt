__lean_models_v2 = []

def __lean_capture_v2(value):
    __lean_models_v2.append(value)
    return value
LEGACY_OBSERVATION = 'capacity.csv\nCapacity\n520\n\nproducts.csv\nProductName,Value,Weight\nNSAIDs,250,913\nAntirheumatic Drugs,178,754\nAcetic Acid Derivatives,313,428\nAntibiotics,301,711\nAntiviral Drugs,425,350\nAntifungal Agents,260,159\nAntidepressants,848,353\nAntipsychotics,934,291\nAntihistamines,114,302\nCorticosteroids,1357,50\nBeta Blockers,156,250\nCalcium Channel Blockers,1780,178\nACE Inhibitors,695,313\nAngiotensin II Receptor Blockers,405,378\nDiuretics,320,94\nStatins,320,97\nInsulin,1357,470\nAnticoagulants,1357,341\nAntiepileptic Drugs,405,121\nAntiemetics,998,61'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'Capacity': '520'}}, {'source': 'products.csv', 'values': {'ProductName': 'NSAIDs', 'Value': '250', 'Weight': '913'}}, {'source': 'products.csv', 'values': {'ProductName': 'Antirheumatic Drugs', 'Value': '178', 'Weight': '754'}}, {'source': 'products.csv', 'values': {'ProductName': 'Acetic Acid Derivatives', 'Value': '313', 'Weight': '428'}}, {'source': 'products.csv', 'values': {'ProductName': 'Antibiotics', 'Value': '301', 'Weight': '711'}}, {'source': 'products.csv', 'values': {'ProductName': 'Antiviral Drugs', 'Value': '425', 'Weight': '350'}}, {'source': 'products.csv', 'values': {'ProductName': 'Antifungal Agents', 'Value': '260', 'Weight': '159'}}, {'source': 'products.csv', 'values': {'ProductName': 'Antidepressants', 'Value': '848', 'Weight': '353'}}, {'source': 'products.csv', 'values': {'ProductName': 'Antipsychotics', 'Value': '934', 'Weight': '291'}}, {'source': 'products.csv', 'values': {'ProductName': 'Antihistamines', 'Value': '114', 'Weight': '302'}}, {'source': 'products.csv', 'values': {'ProductName': 'Corticosteroids', 'Value': '1357', 'Weight': '50'}}, {'source': 'products.csv', 'values': {'ProductName': 'Beta Blockers', 'Value': '156', 'Weight': '250'}}, {'source': 'products.csv', 'values': {'ProductName': 'Calcium Channel Blockers', 'Value': '1780', 'Weight': '178'}}, {'source': 'products.csv', 'values': {'ProductName': 'ACE Inhibitors', 'Value': '695', 'Weight': '313'}}, {'source': 'products.csv', 'values': {'ProductName': 'Angiotensin II Receptor Blockers', 'Value': '405', 'Weight': '378'}}, {'source': 'products.csv', 'values': {'ProductName': 'Diuretics', 'Value': '320', 'Weight': '94'}}, {'source': 'products.csv', 'values': {'ProductName': 'Statins', 'Value': '320', 'Weight': '97'}}, {'source': 'products.csv', 'values': {'ProductName': 'Insulin', 'Value': '1357', 'Weight': '470'}}, {'source': 'products.csv', 'values': {'ProductName': 'Anticoagulants', 'Value': '1357', 'Weight': '341'}}, {'source': 'products.csv', 'values': {'ProductName': 'Antiepileptic Drugs', 'Value': '405', 'Weight': '121'}}, {'source': 'products.csv', 'values': {'ProductName': 'Antiemetics', 'Value': '998', 'Weight': '61'}}]
from gurobipy import Model, GRB

def solve_pharmacy_optimization(LEGACY_RECORDS):
    products = []
    capacity = None
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'products.csv':
            products.append({'ProductName': rec['values']['ProductName'], 'Value': int(rec['values']['Value']), 'Weight': int(rec['values']['Weight'])})
        elif rec['source'] == 'capacity.csv':
            if 'Capacity' in rec['values']:
                if capacity is not None:
                    raise ValueError('Multiple capacities found in LEGACY_RECORDS')
                capacity = int(rec['values']['Capacity'])
    if capacity is None:
        raise ValueError('No capacity found in LEGACY_RECORDS')
    if len(products) == 0:
        raise ValueError('No products found in LEGACY_RECORDS')
    n = len(products)
    for i, prod in enumerate(products):
        if 'ProductName' not in prod or 'Value' not in prod or 'Weight' not in prod:
            raise ValueError(f'Missing data in product {i + 1}')
    idx = list(range(n))
    product_keys = [products[i]['ProductName'] for i in idx]
    value = {product_keys[i]: products[i]['Value'] for i in idx}
    weight = {product_keys[i]: products[i]['Weight'] for i in idx}
    m = __lean_capture_v2(Model())
    m.Params.MIPGap = 0.0001
    x = m.addVars(product_keys, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(sum((value[k] * x[k] for k in product_keys)), GRB.MAXIMIZE)
    m.addConstr(sum((weight[k] * x[k] for k in product_keys)) <= capacity, name='cap')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for k in product_keys:
            print(f'{x[k].VarName} {x[k].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m