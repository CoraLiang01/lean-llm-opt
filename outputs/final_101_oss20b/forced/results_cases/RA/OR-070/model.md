**Mathematical Model**

\[
\begin{aligned}
\text{minimize}\quad & \sum_{m\in M}\sum_{p\in P} c_{m,p}\,x_{m,p} \\[4pt]
\text{subject to}\quad
& \sum_{p\in P} x_{m,p}=1 && \forall m\in M \quad\text{(each manager gets one project)}\\
& \sum_{m\in M} x_{m,p}=1 && \forall p\in P \quad\text{(each project gets one manager)}\\[4pt]
& x_{m,p}\in\{0,1\} && \forall m\in M,\;p\in P
\end{aligned}
\]

---

**Data Mapping**

| Parameter / Set | Source | Mapping Rule |
|-----------------|--------|--------------|
| \(M\) | `file_0_view_0` | `M = { Manager }` (unique values of the column **Manager**) |
| \(P\) | `file_0_view_0` | `P = {1,2,3,4,5,6,7}` (derived from column names **Project 1 Cost** … **Project 7 Cost**) |
| \(c_{m,p}\) | `file_0_view_0` | `c_{m,p} = value in row where Manager = m, column = "Project p Cost"` |

All other data (objective coefficients, constraints) are derived directly from these mappings.