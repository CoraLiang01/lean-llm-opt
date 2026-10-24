**Linear‑programming model**

```
SETS
    P   :  {1,2,…,48}          ! 30‑min periods over 24 h
    S   :  P                    ! possible shift start periods

PARAMETERS
    R[p]   :  required waitstaff at period p   (p∈P)

VARIABLES
    x[s]   :  number of staff whose shift starts at period s   (s∈S)
    x[s]   ∈ ℤ₊

CONSTRAINTS
    ! Coverage: each period must be staffed at least as required
    ∀p∈P :  Σ_{s∈S : (p - s + 48) mod 48 ≤ 15}  x[s]  ≥  R[p]

OBJECTIVE
    minimize   Σ_{s∈S}  x[s]          ! total staff employed
```

*Explanation of the coverage constraint*  
A shift that starts at period `s` works continuously for 8 h = 16 half‑hour periods, i.e. it covers periods  
`s, s+1, …, s+15` (indices taken modulo 48).  
Thus period `p` is covered by a shift starting at `s` iff  
`(p - s + 48) mod 48 ≤ 15`.  
The sum over all such `s` gives the total staff present at period `p`.

---

**Data Mapping**

| Table | Column | Mapping rule |
|-------|--------|--------------|
| `file_0_view_0` | `Requirement` | For each returned row `i` (1‑based, in the order shown in the observation), set `R[i]` to the integer value of the `Requirement` column. |
| | `Time` | Ignored for the model; only the row order defines the period index. |

All other parameters are derived from the table; no additional data are required. The model uses only the sets, parameters, and variables defined above and respects the integer, non‑negativity, and coverage constraints implied by the problem statement.