import gurobipy as gp
from gurobipy import GRB
workers = ['Carpenter', 'Electrician', 'Painter', 'Worker_004', 'Worker_005', 'Worker_006', 'Worker_007', 'Worker_008', 'Worker_009', 'Worker_010', 'Worker_011', 'Worker_012', 'Worker_013', 'Worker_014', 'Worker_015', 'Worker_016', 'Worker_017', 'Worker_018', 'Worker_019', 'Worker_020', 'Worker_021', 'Worker_022', 'Worker_023', 'Worker_024', 'Worker_025', 'Worker_026', 'Worker_027', 'Worker_028', 'Worker_029', 'Worker_030', 'Worker_031', 'Worker_032', 'Worker_033', 'Worker_034', 'Worker_035', 'Worker_036', 'Worker_037', 'Worker_038', 'Worker_039', 'Worker_040', 'Worker_041', 'Worker_042', 'Worker_043', 'Worker_044', 'Worker_045', 'Worker_046', 'Worker_047', 'Worker_048', 'Worker_049', 'Worker_050', 'Worker_051', 'Worker_052', 'Worker_053', 'Worker_054', 'Worker_055', 'Worker_056', 'Worker_057', 'Worker_058', 'Worker_059', 'Worker_060', 'Worker_061', 'Worker_062', 'Worker_063', 'Worker_064', 'Worker_065', 'Worker_066', 'Worker_067', 'Worker_068', 'Worker_069', 'Worker_070', 'Worker_071', 'Worker_072', 'Worker_073', 'Worker_074', 'Worker_075', 'Worker_076', 'Worker_077', 'Worker_078', 'Worker_079', 'Worker_080', 'Worker_081', 'Worker_082', 'Worker_083', 'Worker_084', 'Worker_085', 'Worker_086', 'Worker_087', 'Worker_088', 'Worker_089', 'Worker_090', 'Worker_091', 'Worker_092', 'Worker_093', 'Worker_094', 'Worker_095', 'Worker_096', 'Worker_097', 'Worker_098', 'Worker_099', 'Worker_100', 'Worker_101', 'Worker_102', 'Worker_103', 'Worker_104', 'Worker_105', 'Worker_106', 'Worker_107', 'Worker_108', 'Worker_109', 'Worker_110', 'Worker_111', 'Worker_112', 'Worker_113', 'Worker_114', 'Worker_115', 'Worker_116', 'Worker_117', 'Worker_118', 'Worker_119', 'Worker_120', 'Worker_121', 'Worker_122', 'Worker_123', 'Worker_124', 'Worker_125', 'Worker_126', 'Worker_127', 'Worker_128', 'Worker_129', 'Worker_130', 'Worker_131', 'Worker_132', 'Worker_133', 'Worker_134', 'Worker_135', 'Worker_136', 'Worker_137', 'Worker_138', 'Worker_139', 'Worker_140', 'Worker_141', 'Worker_142', 'Worker_143', 'Worker_144', 'Worker_145', 'Worker_146', 'Worker_147', 'Worker_148', 'Worker_149', 'Worker_150']
homeowners = workers.copy()
N = len(workers)
D = {'Carpenter': {'Carpenter': 2, 'Electrician': 1, 'Painter': 1, 'Worker_004': 3, 'Worker_005': 3}, 'Electrician': {'Carpenter': 2, 'Electrician': 2, 'Painter': 2, 'Worker_004': 2, 'Worker_005': 2}, 'Painter': {'Carpenter': 1, 'Electrician': 2, 'Painter': 3, 'Worker_004': 2, 'Worker_005': 2}, 'Worker_004': {'Carpenter': 3, 'Electrician': 2, 'Painter': 2, 'Worker_004': 1, 'Worker_005': 2}, 'Worker_005': {'Carpenter': 2, 'Electrician': 3, 'Painter': 2, 'Worker_004': 2, 'Worker_005': 1}}
for h in homeowners:
    if h not in D:
        D[h] = {}
    for w in workers:
        if w not in D[h]:
            D[h][w] = 0
for h in homeowners:
    if h not in D or not isinstance(D[h], dict):
        raise ValueError(f'Missing D row for homeowner {h}')
    for w in workers:
        if w not in D[h]:
            raise ValueError(f'Missing D[{h}][{w}]')
m = gp.Model('mutual_wage_balance')
m.Params.MIPGap = 0.0001
w = m.addVars(workers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.addConstr(w['Carpenter'] == 60.0, name='wage_norm')
for i in workers:
    lhs = gp.quicksum((D[k][i] * w[i] for k in workers if k != i))
    rhs = gp.quicksum((D[i][j] * w[j] for j in workers if j != i))
    m.addConstr(lhs == rhs, name=f'balance_{i}')
m.setObjective(gp.quicksum((w[j] for j in workers)), GRB.MINIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for j in workers:
        print(f'w[{j}]: {w[j].X}')
else:
    print(f'Solver status: {m.Status}')