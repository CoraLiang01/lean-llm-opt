**Model**

```
SETS
    M   :  {MA, MB, MC, MD, ME, MF}          ! managers
    P   :  {P1, P2, P3, P4, P5, P6}          ! projects

PARAMETERS
    c[m,p]  :  cost of assigning manager m to project p   (m∈M, p∈P)

VARIABLES
    x[m,p]  :  binary decision variable
               1  if manager m is assigned to project p
               0  otherwise

MINIMIZE
    Z = Σ_{m∈M} Σ_{p∈P} c[m,p] * x[m,p]

CONSTRAINTS
    ! each manager is assigned to exactly one project
    ∀ m∈M :  Σ_{p∈P} x[m,p] = 1

    ! each project is assigned to exactly one manager
    ∀ p∈P :  Σ_{m∈M} x[m,p] = 1

    ! binary variables
    ∀ m∈M, p∈P :  x[m,p] ∈ {0,1}
```

**Data Mapping**

```
TABLE file_0_view_0
    Column "Unnamed: 0"  →  manager index m ∈ M
    Columns "P1" … "P6"  →  project index p ∈ P

    For each (m,p):
        c[m,p] ← value in the cell at row where "Unnamed: 0" = m
                 and column = p
```

The sets `M` and `P` are derived from the distinct values in the column `"Unnamed: 0"` and the column names `"P1"`–`"P6"` of the CSV file, respectively. No numerical values are enumerated in the model; all parameters are mapped directly from the source table.