**Mathematical Model**

```
Sets
  S   – schools  {I, II}
  N   – neighborhoods {N01,…,N31}
  R   – races {w, nw}

Parameters
  Cap[s]          – capacity of school s   (file_0_view_0, column Capacity)
  PopW[n]         – white population in neighborhood n   (file_1_view_0, column Population_White)
  PopNW[n]        – non‑white population in neighborhood n (file_1_view_0, column Population_NonWhite)
  Dist[s,n]       – distance (miles) from school s to neighborhood n
                    (file_2_view_0, row School, column Nxx)

Decision Variables
  x[s,n,r]  ≥ 0   – number of students of race r from neighborhood n assigned to school s

Objective
  minimize  Σ_{s∈S} Σ_{n∈N} Dist[s,n] · ( x[s,n,w] + x[s,n,nw] )

Constraints
  {Neighborhood population}
  ∀ n∈N:
      Σ_{s∈S} x[s,n,w]   = PopW[n]          (1)
      Σ_{s∈S} x[s,n,nw]  = PopNW[n]         (2)

  {School capacity}
  ∀ s∈S:
      Σ_{n∈N} ( x[s,n,w] + x[s,n,nw] ) ≤ Cap[s]   (3)

  {Racial balance – 60 % white ±10 % → 50 %–70 %}
  ∀ s∈S:
      0.5 · Σ_{n∈N} ( x[s,n,w] + x[s,n,nw] ) ≤ Σ_{n∈N} x[s,n,w]   (4a)
      Σ_{n∈N} x[s,n,w] ≤ 0.7 · Σ_{n∈N} ( x[s,n,w] + x[s,n,nw] )   (4b)

  {Non‑negativity}
  ∀ s∈S, n∈N, r∈R:   x[s,n,r] ≥ 0
```

All constraints are linear; the model is a linear program (continuous variables).  
If integer solutions are required, declare `x[s,n,r]` as integer.

---

**Data‑Mapping Rules**

| Parameter | Source Table | Column(s) | Indexing |
|-----------|--------------|-----------|----------|
| `S` | `file_0_view_0` | `School` | each distinct value |
| `N` | `file_1_view_0` | `Neighborhood` | each distinct value |
| `Cap[s]` | `file_0_view_0` | `Capacity` | indexed by `School` |
| `PopW[n]` | `file_1_view_0` | `Population_White` | indexed by `Neighborhood` |
| `PopNW[n]` | `file_1_view_0` | `Population_NonWhite` | indexed by `Neighborhood` |
| `Dist[s,n]` | `file_2_view_0` | matrix columns `N01`–`N31` | row key `School`, column key `Neighborhood` |

- The matrix `file_2_view_0` is a 2 × 31 table; each row is a school, each column a neighborhood.  
- All identifiers (`I`, `II`, `N01`–`N31`) are taken verbatim from the source tables; no additional filtering or renaming is applied.  
- The racial‑balance constraints use the district white ratio 60 % with a ±10 % tolerance, implemented as the bounds 50 %–70 % in constraints (4a)–(4b).  

No other data or parameters are introduced beyond those explicitly listed above.