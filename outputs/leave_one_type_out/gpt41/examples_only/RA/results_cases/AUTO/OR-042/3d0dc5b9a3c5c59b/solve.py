LEGACY_OBSERVATION = 'capacity.csv\nCapacity\n4120\n\nproducts.csv\nProductName,Value,Weight\nNSAIDs,585,50\nAntirheumatic Drugs,557,329\nAcetic Acid Derivatives,963,410\nAntibiotics,301,452\nAntiviral Drugs,425,350\nAntifungal Agents,260,159\nAntidepressants,848,353\nAntipsychotics,461,291\nAntihistamines,840,302\nCorticosteroids,999,50\nBeta Blockers,392,250\nCalcium Channel Blockers,874,178\nACE Inhibitors,695,313\nAngiotensin II Receptor Blockers,405,378\nDiuretics,320,94\nStatins,913,97\nInsulin,754,470\nAnticoagulants,428,341\nAntiepileptic Drugs,711,121\nAntiemetics,998,61'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'Capacity': '4120'}}, {'source': 'products.csv', 'values': {'ProductName': 'NSAIDs', 'Value': '585', 'Weight': '50'}}, {'source': 'products.csv', 'values': {'ProductName': 'Antirheumatic Drugs', 'Value': '557', 'Weight': '329'}}, {'source': 'products.csv', 'values': {'ProductName': 'Acetic Acid Derivatives', 'Value': '963', 'Weight': '410'}}, {'source': 'products.csv', 'values': {'ProductName': 'Antibiotics', 'Value': '301', 'Weight': '452'}}, {'source': 'products.csv', 'values': {'ProductName': 'Antiviral Drugs', 'Value': '425', 'Weight': '350'}}, {'source': 'products.csv', 'values': {'ProductName': 'Antifungal Agents', 'Value': '260', 'Weight': '159'}}, {'source': 'products.csv', 'values': {'ProductName': 'Antidepressants', 'Value': '848', 'Weight': '353'}}, {'source': 'products.csv', 'values': {'ProductName': 'Antipsychotics', 'Value': '461', 'Weight': '291'}}, {'source': 'products.csv', 'values': {'ProductName': 'Antihistamines', 'Value': '840', 'Weight': '302'}}, {'source': 'products.csv', 'values': {'ProductName': 'Corticosteroids', 'Value': '999', 'Weight': '50'}}, {'source': 'products.csv', 'values': {'ProductName': 'Beta Blockers', 'Value': '392', 'Weight': '250'}}, {'source': 'products.csv', 'values': {'ProductName': 'Calcium Channel Blockers', 'Value': '874', 'Weight': '178'}}, {'source': 'products.csv', 'values': {'ProductName': 'ACE Inhibitors', 'Value': '695', 'Weight': '313'}}, {'source': 'products.csv', 'values': {'ProductName': 'Angiotensin II Receptor Blockers', 'Value': '405', 'Weight': '378'}}, {'source': 'products.csv', 'values': {'ProductName': 'Diuretics', 'Value': '320', 'Weight': '94'}}, {'source': 'products.csv', 'values': {'ProductName': 'Statins', 'Value': '913', 'Weight': '97'}}, {'source': 'products.csv', 'values': {'ProductName': 'Insulin', 'Value': '754', 'Weight': '470'}}, {'source': 'products.csv', 'values': {'ProductName': 'Anticoagulants', 'Value': '428', 'Weight': '341'}}, {'source': 'products.csv', 'values': {'ProductName': 'Antiepileptic Drugs', 'Value': '711', 'Weight': '121'}}, {'source': 'products.csv', 'values': {'ProductName': 'Antiemetics', 'Value': '998', 'Weight': '61'}}]
from gurobipy import Model, GRB

def solve_pharmacy_restocking(LEGACY_RECORDS):
    capacities = [int(r['values']['Capacity']) for r in LEGACY_RECORDS if r['source'] == 'capacity.csv']
    if not capacities:
        raise ValueError('No capacity found in LEGACY_RECORDS')
    total_capacity = sum(capacities)
    products = [r for r in LEGACY_RECORDS if r['source'] == 'products.csv']
    if not products:
        raise ValueError('No products found in LEGACY_RECORDS')
    product_keys = []
    values = {}
    weights = {}
    for p in products:
        name = p['values']['ProductName']
        product_keys.append(name)
        try:
            values[name] = int(p['values']['Value'])
            weights[name] = int(p['values']['Weight'])
        except Exception as e:
            raise ValueError(f'Invalid Value/Weight for product {name}: {e}')
    if set(values.keys()) != set(product_keys) or set(weights.keys()) != set(product_keys):
        raise ValueError('Mismatch in product keys and coefficients')
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(product_keys, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(sum((values[p] * x[p] for p in product_keys)), GRB.MAXIMIZE)
    m.addConstr(sum((weights[p] * x[p] for p in product_keys)) <= total_capacity, name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for p in product_keys:
            print(f'{x[p].VarName}: {x[p].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m