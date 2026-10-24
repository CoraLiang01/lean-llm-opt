**Mathematical Model**

Let  

* \(T=\{1,\dots ,48\}\) – the 48 consecutive half‑hour periods that cover the 24‑hour day.  
* \(S=\{1,\dots ,48\}\) – the set of possible shift start times (one per half‑hour).  
* \(R_t\) – the required number of waitstaff in period \(t\) (obtained from *44.csv*).  
* \(A_{t,s}\in\{0,1\}\) – equals 1 if a staff member who starts at time \(s\) works during period \(t\); otherwise 0.  
  \[
  A_{t,s}= 
  \begin{cases}
  1 & \text{if } (t-s)\bmod 48 \in \{0,1,\dots ,15\}\\
  0 & \text{otherwise}
  \end{cases}
  \]
* Decision variable  
  \[
  y_s \in \mathbb{Z}_{\ge 0}\quad\forall s\in S
  \]
  – number of waitstaff whose shift starts at time \(s\).

---

### Objective
\[
\min \;\; \sum_{s\in S} y_s
\]

### Constraints
1. **Coverage of each period**
   \[
   \sum_{s\in S} A_{t,s}\,y_s \;\ge\; R_t \qquad \forall t\in T
   \]

2. **Non‑negativity and integrality**
   \[
   y_s \in \mathbb{Z}_{\ge 0}\qquad \forall s\in S
   \]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(R_t\) | Requirement per period | `file_0_view_0` | `Requirement` |
| \(A_{t,s}\) | Derived from shift length (8 h = 16 half‑hour slots) | – | – |

*The set \(T\) and \(S\) are implicitly defined by the 48 rows of `file_0_view_0`.  The matrix \(A_{t,s}\) is constructed algorithmically from the shift length and does not come from a CSV file.*