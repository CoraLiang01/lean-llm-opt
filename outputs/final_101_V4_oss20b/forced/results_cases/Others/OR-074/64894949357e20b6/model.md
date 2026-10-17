## Mathematical Model

**Indices**  
\(P=\{1,2,\dots ,48\}\) – half‑hour periods of the day.  
\(t\in P\) – possible start period of a waitstaff shift.

**Parameters**  
\(R_p\) – minimum number of waitstaff required in period \(p\)  
  \(R_p =\) integer value of **Requirement** in row \(p\) of *44.csv*.

**Decision Variables**  
\(S_t \ge 0\) – number of waitstaff whose shift starts at period \(t\).

**Objective**  
\[
\min \;\sum_{t\in P} S_t
\]

**Constraints**  

1. **Coverage of each period**  
\[
\sum_{t=\max(1,p-15)}^{p} S_t \;\ge\; R_p
\qquad \forall\, p\in P
\]

2. **Non‑negativity**  
\[
S_t \;\ge\; 0 \qquad \forall\, t\in P
\]

---

## Data Mapping

| Table ID | Column | Mapping |
|----------|--------|---------|
| `file_0_view_0` | `Requirement` | \(R_p = \text{integer value of Requirement in row } p\) |
| `file_0_view_0` | `Time` | *not used in the model* |

The index set \(P\) is derived from the 48 rows of *44.csv*.  
The shift length is fixed at 16 half‑hour periods (8 h), so the lower bound in the coverage constraint is \(\max(1,p-15)\).