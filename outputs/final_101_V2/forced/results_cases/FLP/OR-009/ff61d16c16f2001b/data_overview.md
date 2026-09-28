**Retrieved Data**

---

### customer_demand.csv

| customer | demand |
|----------|--------|
| C1       | 94     |
| C2       | 39     |
| C3       | 65     |
| C4       | 435    |

**Source positions:**
- Row 1: C1, 94
- Row 2: C2, 39
- Row 3: C3, 65
- Row 4: C4, 435

---

### supply_capacity.csv

| Facility (Unnamed: 0) | supply_capacity |
|-----------------------|----------------|
| S1                    | 2531           |
| S2                    | 20             |
| S3                    | 210            |
| S4                    | 241            |

**Source positions:**
- Row 1: S1, 2531
- Row 2: S2, 20
- Row 3: S3, 210
- Row 4: S4, 241

---

### transportation_costs.csv

| Facility (Unnamed: 0) | C1                  | C2                  | C3                  | C4                  |
|-----------------------|---------------------|---------------------|---------------------|---------------------|
| S1                    | 543.756480860856    | 23.685276141764653  | 23.676386730773032  | 447.75143678673766  |
| S2                    | 883.9151090405642   | 0.04977684765576961 | 0.0350986687216299  | 44.45588531711622   |
| S3                    | 537.3456896658107   | 23.769274659075112  | 498.95659249465467  | 440.60737890439776  |
| S4                    | 1791.493192397229   | 68.21633865655126   | 1432.4837339656747  | 1527.7635425462734  |

**Source positions:**
- Row 1: S1, [C1, C2, C3, C4]
- Row 2: S2, [C1, C2, C3, C4]
- Row 3: S3, [C1, C2, C3, C4]
- Row 4: S4, [C1, C2, C3, C4]

---

**All facility IDs, customer IDs, capacities, demands, and cost-matrix axes are preserved with their original source-row positions and orientation. No data has been transposed, truncated, or inferred beyond the provided context.**