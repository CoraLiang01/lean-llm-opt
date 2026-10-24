**Mathematical Model**

```
SETS
    M   : set of managers
    P   : set of projects

PARAMETERS
    c[m,p]   : assignment cost of manager m to project p   (m ∈ M, p ∈ P)

VARIABLES
    x[m,p]   : 1 if manager m is assigned to project p, 0 otherwise   (m ∈ M, p ∈ P)

MINIMIZE
    Z = Σ_{m∈M} Σ_{p∈P} c[m,p] * x[m,p]

CONSTRAINTS
    (* each manager is assigned to exactly one project *)
    forall m in M:
        Σ_{p∈P} x[m,p] = 1

    (* each project receives exactly one manager *)
    forall p in P:
        Σ_{m∈M} x[m,p] = 1

    (* binary decision variables *)
    forall m in M, p in P:
        x[m,p] ∈ {0,1}
```

**Data Mapping**

```
TABLE: file_0_view_0
COLUMNS: 
    Manager,
    Project 1 Cost,
    Project 2 Cost,
    Project 3 Cost,
    Project 4 Cost,
    Project 5 Cost,
    Project 6 Cost,
    Project 7 Cost,
    Project 8 Cost,
    Project 9 Cost,
    Project 10 Cost,
    Project 11 Cost

SET M:
    { Manager 1, Manager 2, Manager 3, Manager 4, Manager 5,
      Manager 6, Manager 7, Manager 8, Manager 9, Manager 10, Manager 11 }

SET P:
    { Project 1, Project 2, Project 3, Project 4, Project 5,
      Project 6, Project 7, Project 8, Project 9, Project 10, Project 11 }

PARAMETER c[m,p]:
    For each row r in the table,
        let m = r["Manager"]
        for each project number k = 1..11:
            let p = "Project " + k
            c[m,p] = r["Project " + k + " Cost"]
```

The model above defines a classic one‑to‑one assignment problem with the cost matrix extracted directly from the CSV file.