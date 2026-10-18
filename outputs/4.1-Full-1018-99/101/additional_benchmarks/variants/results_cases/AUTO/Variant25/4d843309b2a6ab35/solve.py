LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant25/inputs/batch_03/export_03.csv",\n    "values": {\n      "worker_id": "W11",\n      "P00": "1091",\n      "P01": "324",\n      "P02": "379",\n      "P03": "272",\n      "P04": "",\n      "P05": ""\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant25/inputs/batch_03/export_03.csv",\n    "values": {\n      "worker_id": "W10",\n      "P00": "1176",\n      "P01": "1111",\n      "P02": "1380",\n      "P03": "542",\n      "P04": "158",\n      "P05": "922"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant25/inputs/batch_03/export_03.csv",\n    "values": {\n      "worker_id": "W06",\n      "P00": "",\n      "P01": "822",\n      "P02": "",\n      "P03": "223",\n      "P04": "1155",\n      "P05": "1055"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant25/inputs/batch_03/export_03.csv",\n    "values": {\n      "worker_id": "W01",\n      "P00": "5",\n      "P01": "",\n      "P02": "20",\n      "P03": "",\n      "P04": "18",\n      "P05": "7"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant25/inputs/batch_03/export_03.csv",\n    "values": {\n      "worker_id": "W02",\n      "P00": "",\n      "P01": "1397",\n      "P02": "953",\n      "P03": "714",\n      "P04": "205",\n      "P05": ""\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant25/inputs/batch_03/export_03.csv",\n    "values": {\n      "worker_id": "W00",\n      "P00": "1063",\n      "P01": "219",\n      "P02": "",\n      "P03": "1329",\n      "P04": "",\n      "P05": "436"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant25/inputs/batch_03/export_03.csv",\n    "values": {\n      "worker_id": "W04",\n      "P00": "642",\n      "P01": "",\n      "P02": "1130",\n      "P03": "133",\n      "P04": "199",\n      "P05": "311"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant25/inputs/batch_03/export_03.csv",\n    "values": {\n      "worker_id": "W12",\n      "P00": "102",\n      "P01": "353",\n      "P02": "651",\n      "P03": "102",\n      "P04": "",\n      "P05": "7"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant25/inputs/batch_03/export_03.csv', 'values': {'worker_id': 'W11', 'P00': '1091', 'P01': '324', 'P02': '379', 'P03': '272', 'P04': '', 'P05': ''}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant25/inputs/batch_03/export_03.csv', 'values': {'worker_id': 'W10', 'P00': '1176', 'P01': '1111', 'P02': '1380', 'P03': '542', 'P04': '158', 'P05': '922'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant25/inputs/batch_03/export_03.csv', 'values': {'worker_id': 'W06', 'P00': '', 'P01': '822', 'P02': '', 'P03': '223', 'P04': '1155', 'P05': '1055'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant25/inputs/batch_03/export_03.csv', 'values': {'worker_id': 'W01', 'P00': '5', 'P01': '', 'P02': '20', 'P03': '', 'P04': '18', 'P05': '7'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant25/inputs/batch_03/export_03.csv', 'values': {'worker_id': 'W02', 'P00': '', 'P01': '1397', 'P02': '953', 'P03': '714', 'P04': '205', 'P05': ''}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant25/inputs/batch_03/export_03.csv', 'values': {'worker_id': 'W00', 'P00': '1063', 'P01': '219', 'P02': '', 'P03': '1329', 'P04': '', 'P05': '436'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant25/inputs/batch_03/export_03.csv', 'values': {'worker_id': 'W04', 'P00': '642', 'P01': '', 'P02': '1130', 'P03': '133', 'P04': '199', 'P05': '311'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant25/inputs/batch_03/export_03.csv', 'values': {'worker_id': 'W12', 'P00': '102', 'P01': '353', 'P02': '651', 'P03': '102', 'P04': '', 'P05': '7'}}]
import gurobipy as gp
from gurobipy import GRB
skill_order = {'Junior': 0, 'Intermediate': 1, 'Senior': 2, 'Expert': 3}
worker_skill = {'W12': 'Junior', 'W06': 'Expert', 'W04': 'Intermediate', 'W11': 'Junior', 'W10': 'Intermediate', 'W02': 'Intermediate', 'W00': 'Expert'}
worker_on_leave = {'W12': 0, 'W06': 0, 'W04': 0, 'W11': 0, 'W10': 0, 'W02': 0, 'W00': 0}
project_required_skill = {'P00': 'Junior', 'P01': 'Junior', 'P02': 'Junior', 'P03': 'Junior', 'P04': 'Junior', 'P05': 'Intermediate'}
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant25/inputs/batch_03/export_03.csv', 'values': {'worker_id': 'W11', 'P00': '1091', 'P01': '324', 'P02': '379', 'P03': '272', 'P04': '', 'P05': ''}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant25/inputs/batch_03/export_03.csv', 'values': {'worker_id': 'W10', 'P00': '1176', 'P01': '1111', 'P02': '1380', 'P03': '542', 'P04': '158', 'P05': '922'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant25/inputs/batch_03/export_03.csv', 'values': {'worker_id': 'W06', 'P00': '', 'P01': '822', 'P02': '', 'P03': '223', 'P04': '1155', 'P05': '1055'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant25/inputs/batch_03/export_03.csv', 'values': {'worker_id': 'W01', 'P00': '5', 'P01': '', 'P02': '20', 'P03': '', 'P04': '18', 'P05': '7'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant25/inputs/batch_03/export_03.csv', 'values': {'worker_id': 'W02', 'P00': '', 'P01': '1397', 'P02': '953', 'P03': '714', 'P04': '205', 'P05': ''}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant25/inputs/batch_03/export_03.csv', 'values': {'worker_id': 'W00', 'P00': '1063', 'P01': '219', 'P02': '', 'P03': '1329', 'P04': '', 'P05': '436'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant25/inputs/batch_03/export_03.csv', 'values': {'worker_id': 'W04', 'P00': '642', 'P01': '', 'P02': '1130', 'P03': '133', 'P04': '199', 'P05': '311'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant25/inputs/batch_03/export_03.csv', 'values': {'worker_id': 'W12', 'P00': '102', 'P01': '353', 'P02': '651', 'P03': '102', 'P04': '', 'P05': '7'}}]
workers = list(worker_skill.keys())
projects = list(project_required_skill.keys())
cost = {}
eligibility_mask = {}
for rec in LEGACY_RECORDS:
    w = rec['values']['worker_id']
    if w not in workers:
        continue
    cost[w] = {}
    eligibility_mask[w] = {}
    for p in projects:
        val = rec['values'].get(p, '')
        eligible = worker_on_leave[w] == 0 and skill_order[worker_skill[w]] >= skill_order[project_required_skill[p]] and (val != '')
        eligibility_mask[w][p] = 1 if eligible else 0
        if eligible:
            cost[w][p] = int(val)
E = [(w, p) for w in workers for p in projects if eligibility_mask[w][p] == 1]
for w, p in E:
    if w not in cost or p not in cost[w]:
        raise ValueError(f'Missing cost for eligible pair ({w},{p})')
m = gp.Model('AP')
x_vars = m.addVars(E, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[w][p] * x_vars[w, p] for w, p in E)), GRB.MINIMIZE)
for p in projects:
    eligible_ws = [w for w in workers if eligibility_mask[w][p] == 1]
    m.addConstr(gp.quicksum((x_vars[w, p] for w in eligible_ws)) == 1, name='prj_' + p)
for w in workers:
    eligible_ps = [p for p in projects if eligibility_mask[w][p] == 1]
    m.addConstr(gp.quicksum((x_vars[w, p] for p in eligible_ps)) <= 1, name='wrk_' + w)
for w in workers:
    for p in projects:
        if eligibility_mask[w][p] == 0:
            continue
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')