**Mathematical Model**

```
SETS
    W   :  {1,2,3}                     ! workstations
    M   :  {1,…,101}                   ! radio models

PARAMETERS
    t[w,m]   ≥ 0   : processing time (minutes) of model m on workstation w
    cap[w]   ≥ 0   : effective daily capacity of workstation w

VARIABLES
    x[m]     ∈ ℤ₊   : number of units of model m produced per day
    idle[w]  ≥ 0    : idle time (minutes) of workstation w

MINIMIZE
    Σ_{w∈W} idle[w]                           ! total idle time

CONSTRAINTS
    1.  Σ_{m∈M} t[w,m] * x[m]  ≤  cap[w]      ∀ w∈W   ! capacity limits
    2.  idle[w]  =  cap[w]  –  Σ_{m∈M} t[w,m] * x[m]   ∀ w∈W   ! idle definition
```

---

**Data Mapping Rules**

| Parameter | Source Table | Column | Row Filter | Expression |
|-----------|---------------|--------|------------|------------|
| `t[w,m]`  | `file_0_view_0` | `"HiFi{m}_Minutes"` | `Workstation = w` | value |
| `maintenance_percent[w]` | `file_0_view_0` | `"Maintenance_Percent"` | `Workstation = w` | value |
| `cap[w]` | – | – | – | `1440 * (1 - maintenance_percent[w]/100)` |

*Notes*

- `W` is the set of distinct values in the `Workstation` column of the table.  
- `M` is the set of integers 1 through 101, corresponding to the columns `HiFi1_Minutes` … `HiFi101_Minutes`.  
- All numeric columns in the table are treated as real parameters; only the production quantities `x[m]` are integer variables.  
- The objective and constraints are expressed symbolically; no numerical values are hard‑coded.