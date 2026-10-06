Here is the complete offer data for all available worker/project pairings, including each worker's on_leave status, skill, and each project's required_skill:

---

### Workers (with skill and on_leave status)
| worker_id | skill        | on_leave |
|-----------|--------------|----------|
| W01       | Senior       | 1        |
| W00       | Junior       | 0        |
| W19       | Junior       | 0        |
| W11       | Senior       | 0        |
| W04       | Junior       | 0        |
| W15       | Expert       | 0        |
| W10       | Expert       | 0        |
| W06       | Expert       | 0        |
| W20       | Intermediate | 0        |
| W14       | Intermediate | 0        |

---

### Projects (with required_skill)
| project_id | required_skill |
|------------|---------------|
| P00        | Intermediate  |
| P01        | Intermediate  |
| P02        | Senior        |
| P03        | Junior        |
| P04        | Junior        |
| P05        | Junior        |
| P06        | Junior        |
| P07        | Junior        |

---

### Offer List (worker_id, project_id, cost_cents, on_leave, worker_skill, required_skill)

#### P00 (Intermediate)
- W10, 147, 0, Expert, Intermediate
- W19, 5, 0, Junior, Intermediate
- W11, 235, 0, Senior, Intermediate
- W20, 675, 0, Intermediate, Intermediate
- W04, 4, 0, Junior, Intermediate
- W01, 19, 1, Senior, Intermediate
- W06, 1217, 0, Expert, Intermediate
- W15, 722, 0, Expert, Intermediate

#### P01 (Intermediate)
- W10, 114, 0, Expert, Intermediate
- W00, 17, 0, Junior, Intermediate
- W20, 1055, 0, Intermediate, Intermediate
- W04, 8, 0, Junior, Intermediate
- W06, 425, 0, Expert, Intermediate
- W14, 1361, 0, Intermediate, Intermediate

#### P02 (Senior)
- W19, 9, 0, Junior, Senior
- W11, 1034, 0, Senior, Senior
- W20, 8, 0, Intermediate, Senior
- W04, 16, 0, Junior, Senior
- W01, 6, 1, Senior, Senior
- W06, 1320, 0, Expert, Senior
- W15, 577, 0, Expert, Senior

#### P03 (Junior)
- W10, 998, 0, Expert, Junior
- W19, 109, 0, Junior, Junior
- W11, 782, 0, Senior, Junior
- W20, 703, 0, Intermediate, Junior
- W04, 543, 0, Junior, Junior
- W06, 1097, 0, Expert, Junior
- W15, 1062, 0, Expert, Junior
- W14, 379, 0, Intermediate, Junior

#### P04 (Junior)
- W10, 1270, 0, Expert, Junior
- W19, 1306, 0, Junior, Junior
- W11, 556, 0, Senior, Junior
- W00, 430, 0, Junior, Junior
- W04, 205, 0, Junior, Junior
- W06, 822, 0, Expert, Junior
- W15, 1314, 0, Expert, Junior
- W14, 239, 0, Intermediate, Junior

#### P05 (Junior)
- W19, 626, 0, Junior, Junior
- W11, 1018, 0, Senior, Junior
- W00, 1385, 0, Junior, Junior
- W04, 1149, 0, Junior, Junior
- W01, 19, 1, Senior, Junior
- W06, 221, 0, Expert, Junior
- W15, 528, 0, Expert, Junior
- W14, 460, 0, Intermediate, Junior

#### P06 (Junior)
- W19, 466, 0, Junior, Junior
- W00, 260, 0, Junior, Junior
- W20, 893, 0, Intermediate, Junior
- W04, 533, 0, Junior, Junior
- W01, 8, 1, Senior, Junior
- W06, 496, 0, Expert, Junior
- W15, 835, 0, Expert, Junior
- W14, 130, 0, Intermediate, Junior

#### P07 (Junior)
- W10, 953, 0, Expert, Junior
- W19, 1365, 0, Junior, Junior
- W11, 1136, 0, Senior, Junior
- W00, 1107, 0, Junior, Junior
- W20, 129, 0, Intermediate, Junior
- W04, 732, 0, Junior, Junior
- W01, 13, 1, Senior, Junior
- W06, 908, 0, Expert, Junior
- W15, 918, 0, Expert, Junior

---

**Note:**  
- Only workers with `on_leave=0` are eligible.
- A worker's skill must be **at least** the required_skill for the project, with the hierarchy: Junior < Intermediate < Senior < Expert.
- Only the listed offers are permitted.

This is the complete data as requested.