**Mathematical Model**

```
Indices
-------
I  = {1,2,…,110}          // all Project IDs in project.csv

Parameters
----------
Capital[i]   ∈ ℝ⁺   for i∈I   // capital cost (k$) – column “Capital (k$)”
NPV[i]      ∈ ℝ⁺   for i∈I   // expected NPV (k$) – column “NPV (k$)”

Decision Variables
-------------------
x[i] ∈ {0,1}   for i∈I       // 1 if project i is selected, 0 otherwise

Objective
---------
max  Σ_{i∈I}  NPV[i] · x[i]          // maximize total expected NPV

Constraints
-----------
// Budget
Σ_{i∈I}  Capital[i] · x[i]  ≤ 1000

// Mutual exclusivity: Projects 4 and 7
x[4] + x[7]  ≤ 1

// Pre‑requisite: Project 6 requires Project 1
x[6]  ≤  x[1]

// Contingent: Project 10 requires Project 5
x[10] ≤  x[5]
```

---

**Data Mapping Rules**

| Parameter / Variable | Table ID          | Column Name          | Index Set |
|----------------------|-------------------|----------------------|-----------|
| `Capital[i]`         | `file_0_view_0`   | `Capital (k$)`       | `I`       |
| `NPV[i]`             | `file_0_view_0`   | `NPV (k$)`           | `I`       |
| `x[i]` (decision)    | –                 | –                    | `I`       |

*The index set `I` is the set of all `Project ID` values returned in `project.csv` (1–110).  All parameters are taken directly from the corresponding columns of the single source table `file_0_view_0`.  No additional data or constraints are introduced beyond those specified in the query.*