**Symbolic Linear Programming Model**

Let  

* \(T=\{1,\dots ,48\}\) – indices of the 48 consecutive half‑hour time slots that cover the 24‑hour day.  
* \(S=\{1,\dots ,48\}\) – indices of possible start times for a waitstaff shift (a shift that starts at slot \(s\) covers slots \(s,s+1,\dots ,s+15\)).  
* \(r_t\) – required number of waitstaff in slot \(t\).  
* \(x_s\) – number of waitstaff whose shift starts at slot \(s\).

---

### Objective
\[
\min \;\sum_{s\in S} x_s
\]

### Constraints
For every time slot \(t\in T\) the total number of staff covering that slot must be at least the requirement:
\[
\sum_{\substack{s\in S\\ s\le t < s+16}} x_s \;\ge\; r_t
\qquad \forall\, t\in T
\]

### Variable Domains
\[
x_s \in \mathbb{Z}_{\ge 0}\qquad \forall\, s\in S
\]

---

**Data Mapping**

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(r_t\) | `file_0_view_0` (table_id) | `Requirement` column, row \(t-1\) (0‑based) |
| \(T\) | – | \(\{1,\dots ,48\}\) |
| \(S\) | – | \(\{1,\dots ,48\}\) |

*The table `file_0_view_0` contains 48 rows, each with a `Time` string and a `Requirement` integer. The integer in the `Requirement` column is used directly as \(r_t\).*

---

This LP formulation determines the minimum number of waitstaff needed to satisfy the hourly requirements while respecting the 8‑hour continuous shift constraint.