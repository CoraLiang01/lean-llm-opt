**Data Mapping**

| Table ID | Column | Source |
|----------|--------|--------|
| `file_0_view_0` | `Shift` | integer 1–24 |
| `file_0_view_0` | `Number Required` | required drivers for that hour |

Let  

\[
R_t = \text{Number Required for shift } t \quad (t=1,\dots ,24)
\]

be the parameter extracted from the table.

---

### Index Sets
\[
T = \{1,2,\dots ,24\}
\]

### Decision Variables
\[
x_t \;=\; \text{number of drivers that start working at the beginning of shift } t,
\qquad t\in T
\]

### Objective
\[
\min \;\sum_{t\in T} x_t
\]

### Constraints

For each shift \(t\in T\) the total number of drivers that are on duty during that hour must be at least the required number \(R_t\).  
A driver who starts at shift \(s\) works during shifts \(s,\,s+1,\,s+2,\,s+3\) (indices modulo 24).  
Hence the covering constraint for shift \(t\) is

\[
\sum_{\substack{s\in T\\ (t-s)\bmod 24 \in \{0,1,2,3\}}} x_s \;\ge\; R_t ,
\qquad \forall\, t\in T .
\]

### Variable Domains
\[
x_t \;\in\; \mathbb{Z}_{\ge 0}, \qquad \forall\, t\in T .
\]

---

**Model Summary**

\[
\begin{aligned}
\min\;&\sum_{t\in T} x_t \\
\text{s.t.}\;&\sum_{\substack{s\in T\\ (t-s)\bmod 24 \in \{0,1,2,3\}}} x_s \;\ge\; R_t ,
\quad \forall\, t\in T,\\
&x_t \in \mathbb{Z}_{\ge 0}, \quad \forall\, t\in T .
\end{aligned}
\]

All parameters \(R_t\) are taken directly from the `Number Required` column of `file_0_view_0`.