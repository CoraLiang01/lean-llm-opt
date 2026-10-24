## Linear Programming Model

**Indices**  
\(S=\{1,2,\dots ,24\}\) – shift numbers (1 = 0:00‑1:00, …, 24 = 23:00‑0:00)

**Parameters**  
\(r_s\) – number of drivers/crew required in shift \(s\)  
(see Data Mapping below)

**Decision variables**  
\(x_s \ge 0\) – number of drivers/crew who start work at the beginning of shift \(s\)

**Objective**  
\[
\min \;\sum_{s\in S} x_s
\]

**Constraints**  
For every shift \(t\in S\) the total number of drivers that are on duty during that shift must be at least the required number:

\[
x_t + x_{t-1} + x_{t-2} + x_{t-3} \;\ge\; r_t
\qquad \forall\, t\in S
\]

Indices are taken modulo 24, i.e. \(x_{0}=x_{24}\), \(x_{-1}=x_{23}\), \(x_{-2}=x_{22}\), \(x_{-3}=x_{21}\).

---

## Data Mapping

| Table ID | Column | Parameter | Mapping |
|----------|--------|-----------|---------|
| `file_0_view_0` | `Shift` | \(s\) | 1 → 1, 2 → 2, …, 24 → 24 |
| `file_0_view_0` | `Number Required` | \(r_s\) | \(r_1=20,\; r_2=18,\; \dots,\; r_{24}=25\) |

All other columns (`Time`) are not used in the model. The mapping above fully specifies the parameter values needed for the constraints.