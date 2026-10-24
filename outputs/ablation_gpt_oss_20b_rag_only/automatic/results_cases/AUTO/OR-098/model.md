**Mathematical Model**

```
SETS
    W   : workers  (all column names of the table except the column “Owner”)
    O   : owners   (unique values that appear in the column “Owner”)
          // In this data O = W

PARAMETERS
    a[i,j]   : number of days worker j spent on homeowner i’s home
                (i ∈ O , j ∈ W)   ←  table file_0_view_0
    w0        : fixed daily wage of the first worker (“Carpenter”) = 60.00

VARIABLES
    w[j]      : daily wage of worker j   (j ∈ W)
    w[j] ≥ 0

OBJECTIVE
    minimize  Σ_{j∈W} w[j]          // the constant 60 does not affect the optimum

CONSTRAINTS
    1)  w["Carpenter"] = w0

    2)  For every i ∈ O:
            Σ_{j∈W} a[i,j] * w[j]  –  a[i,i] * w[i]  = 0
```

*Explanation of the balance constraint*  
For homeowner i the left‑hand side is the total income earned by i from working on all
other homes (the term with j ≠ i).  
The right‑hand side is the total expenditure of i for the work performed on his own
home.  Setting the two equal enforces the required mutual‑payment balance.

---

**Data Mapping**

| Parameter / Set | Source | Mapping Details |
|-----------------|--------|-----------------|
| `W` | Table `file_0_view_0` | All column names except the column named `"Owner"` |
| `O` | Table `file_0_view_0` | All distinct values that appear in the column `"Owner"` |
| `a[i,j]` | Table `file_0_view_0` | For each row where `Owner = i`, read the numeric entry in column `j` |
| `w0` | Constant | 60.00 (no table reference) |

All indices used in the constraints (`i ∈ O`, `j ∈ W`) belong to the declared
domains.  The variable `w[j]` is defined for every worker in `W`; the first worker
(`"Carpenter"`) is fixed by constraint 1.  The objective is linear and the model
is a linear feasibility problem with a simple minimizing objective.