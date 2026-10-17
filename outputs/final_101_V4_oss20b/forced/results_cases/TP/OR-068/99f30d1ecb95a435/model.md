**Mathematical Model**

Let  

* \(M=\{MA,MB,MC,MD,ME,MF\}\) – set of managers  
* \(P=\{P1,P2,P3,P4,P5,P6\}\) – set of projects  

Parameters  

* \(c_{mp}\) – assignment cost of manager \(m\) to project \(p\) (non‑negative real)

Decision variables  

* \(x_{mp}\in\{0,1\}\) – 1 if manager \(m\) is assigned to project \(p\), 0 otherwise  

Objective  

\[
\min \sum_{m\in M}\sum_{p\in P} c_{mp}\,x_{mp}
\]

Constraints  

1. **Manager assignment** \(\displaystyle \sum_{p\in P} x_{mp}=1,\quad \forall m\in M\)

2. **Project assignment** \(\displaystyle \sum_{m\in M} x_{mp}=1,\quad \forall p\in P\)

3. **Binary variables** \(x_{mp}\in\{0,1\},\quad \forall m\in M,\;p\in P\)

---

**Data Mapping Rules**

| Table ID | Column | Meaning | Mapping |
|----------|--------|---------|---------|
| `file_0_view_0` | `Unnamed: 0` | Manager identifier | \(m\) |
| `file_0_view_0` | `P1`–`P6` | Cost of assigning manager to project | \(c_{mp}\) |

**Rule**  
For each row in `file_0_view_0`, let `m` be the value in column `Unnamed: 0`.  
For each project column `Pj` (j = 1,…,6), set  

\[
c_{m,Pj} = \text{value in column } Pj \text{ of that row}.
\]

All other data (sets \(M,P\) and binary domain of \(x_{mp}\)) are derived directly from the table identifiers and column names. No additional parameters or constraints are required.