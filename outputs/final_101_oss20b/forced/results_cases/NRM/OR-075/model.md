**Mathematical Model**

```markdown
Sets
-----
I  :  {p | p is a Project ID in the data set}

Parameters
----------
Capital[p]   :  capital required for project p  (k$)   ←  column "Capital (k$)"
NPV[p]      :  expected net present value of project p  (k$)   ←  column "NPV (k$)"

Decision Variables
------------------
x[p] ∈ {0,1}   for all p ∈ I   (1 if project p is selected, 0 otherwise)

Objective
---------
maximize   ∑_{p∈I} NPV[p] · x[p]

Constraints
-----------
1. Budget limit  
   ∑_{p∈I} Capital[p] · x[p]  ≤  1000

2. Mutual exclusivity (Projects 4 and 7)  
   x[4] + x[7]  ≤  1

3. Pre‑requisite (Project 6 requires Project 1)  
   x[6]  ≤  x[1]

4. Contingent (Project 10 requires Project 5)  
   x[10]  ≤  x[5]
```

---

**Data Mapping Rules**

| Source | Table ID          | Column Name          | Mapped To          |
|--------|-------------------|----------------------|--------------------|
| Project data | `file_0_view_0` | `"Project ID"`      | Index set `I`      |
| Project data | `file_0_view_0` | `"Capital (k$)"`    | Parameter `Capital[p]` |
| Project data | `file_0_view_0` | `"NPV (k$)"`        | Parameter `NPV[p]` |

*The mapping uses the value in `"Project ID"` as the key for all parameters and variables.*