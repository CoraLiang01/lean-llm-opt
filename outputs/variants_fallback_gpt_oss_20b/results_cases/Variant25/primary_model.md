**Symbolic Model**

Let  

- \(W\) – set of workers (IDs) that satisfy `on_leave = 0`.  
- \(P\) – set of projects.  

Parameters  

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(\text{skill\_lvl}[w]\) | Integer level of worker \(w\) (Junior = 1, Intermediate = 2, Senior = 3, Expert = 4) | `file_0_view_0` column `skill` → mapping table |
| \(\text{req\_lvl}[p]\) | Integer required level for project \(p\) | `file_1_view_0` column `required_skill` → mapping table |
| \(c[w,p]\) | Assignment cost in USD cents (blank → undefined) | `file_2_view_0` columns `P00`–`P05` |
| \(\text{allowed}[w,p]\) | 1 if worker \(w\) can be assigned to project \(p\) (cost defined **and** skill sufficient), 0 otherwise | derived from \(c[w,p]\) and skill comparison |

Decision variables  

- \(x_{w,p}\in\{0,1\}\) – 1 if worker \(w\) is assigned to project \(p\), 0 otherwise.

**Objective**

\[
\min \sum_{w\in W}\sum_{p\in P} c[w,p]\;x_{w,p}
\]

**Constraints**

1. **Project assignment** – each project gets exactly one worker  

\[
\sum_{w\in W} x_{w,p}=1 \quad \forall p\in P
\]

2. **Worker capacity** – each worker takes at most one project  

\[
\sum_{p\in P} x_{w,p}\le 1 \quad \forall w\in W
\]

3. **Feasibility** – assignments only allowed where \(\text{allowed}[w,p]=1\)  

\[
x_{w,p}\le \text{allowed}[w,p] \quad \forall w\in W,\;p\in P
\]

---

**Data Mapping Rules**

| Parameter | Table ID | Column(s) | Notes |
|-----------|----------|-----------|-------|
| \(\text{skill\_lvl}[w]\) | `file_0_view_0` | `skill` | Map `"Junior"`→1, `"Intermediate"`→2, `"Senior"`→3, `"Expert"`→4 |
| \(\text{req\_lvl}[p]\) | `file_1_view_0` | `required_skill` | Same skill‑level mapping as above |
| \(c[w,p]\) | `file_2_view_0` | `P00`–`P05` | Empty string indicates undefined cost |
| \(\text{allowed}[w,p]\) | Derived | – | 1 if \(c[w,p]\) defined **and** \(\text{skill\_lvl}[w]\ge \text{req\_lvl}[p]\); otherwise 0 |

All sets \(W\) and \(P\) are taken directly from the rows returned in the observation tables. The model uses only these entities and respects the constraints specified in the query.