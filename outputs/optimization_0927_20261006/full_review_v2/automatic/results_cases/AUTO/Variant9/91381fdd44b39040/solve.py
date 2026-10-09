LEGACY_OBSERVATION = '[{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv","values":{"Item":"A","Demand":"24"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv","values":{"Item":"B","Demand":"18"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv","values":{"Item":"C","Demand":"12"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv","values":{"Item":"D","Demand":"10"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv","values":{"Pattern":"P1","A":"4","B":"0","C":"0","D":"0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv","values":{"Pattern":"P2","A":"0","B":"3","C":"0","D":"0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv","values":{"Pattern":"P3","A":"0","B":"0","C":"2","D":"0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv","values":{"Pattern":"P4","A":"0","B":"0","C":"0","D":"2"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv","values":{"Pattern":"P5","A":"2","B":"1","C":"0","D":"0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv","values":{"Pattern":"P6","A":"1","B":"0","C":"1","D":"0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv","values":{"Pattern":"P7","A":"0","B":"1","C":"0","D":"1"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv","values":{"Pattern":"P8","A":"1","B":"1","C":"1","D":"0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv","values":{"Pattern":"P9","A":"2","B":"0","C":"0","D":"1"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv', 'values': {'Item': 'A', 'Demand': '24'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv', 'values': {'Item': 'B', 'Demand': '18'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv', 'values': {'Item': 'C', 'Demand': '12'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv', 'values': {'Item': 'D', 'Demand': '10'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv', 'values': {'Pattern': 'P1', 'A': '4', 'B': '0', 'C': '0', 'D': '0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv', 'values': {'Pattern': 'P2', 'A': '0', 'B': '3', 'C': '0', 'D': '0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv', 'values': {'Pattern': 'P3', 'A': '0', 'B': '0', 'C': '2', 'D': '0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv', 'values': {'Pattern': 'P4', 'A': '0', 'B': '0', 'C': '0', 'D': '2'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv', 'values': {'Pattern': 'P5', 'A': '2', 'B': '1', 'C': '0', 'D': '0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv', 'values': {'Pattern': 'P6', 'A': '1', 'B': '0', 'C': '1', 'D': '0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv', 'values': {'Pattern': 'P7', 'A': '0', 'B': '1', 'C': '0', 'D': '1'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv', 'values': {'Pattern': 'P8', 'A': '1', 'B': '1', 'C': '1', 'D': '0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv', 'values': {'Pattern': 'P9', 'A': '2', 'B': '0', 'C': '0', 'D': '1'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
item_demand_records = [r for r in records if r['source'].endswith('item_demand.csv')]
items = []
item_demand = {}
for rec in item_demand_records:
    item = rec['values']['Item']
    demand = int(rec['values']['Demand'])
    items.append(item)
    item_demand[item] = demand
pattern_records = [r for r in records if r['source'].endswith('cutting_patterns.csv')]
patterns = []
pattern_yield = {}
for rec in pattern_records:
    pattern = rec['values']['Pattern']
    patterns.append(pattern)
    pattern_yield[pattern] = {}
    for item in items:
        pattern_yield[pattern][item] = int(rec['values'][item])
for item in items:
    for pattern in patterns:
        if item not in pattern_yield[pattern]:
            raise ValueError(f'Missing coefficient for item {item} in pattern {pattern}')

def build_cutting_stock_model():
    m = gp.Model('cutting_stock')
    y_vars = m.addVars(patterns, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((y_vars[p] for p in patterns)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((pattern_yield[p][i] * y_vars[p] for p in patterns)) >= item_demand[i] for i in items), name='')
    m.Params.MIPGap = 0.0001
    return m
m = build_cutting_stock_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')