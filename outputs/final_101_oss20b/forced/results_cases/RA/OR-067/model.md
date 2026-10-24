**Mathematical Model**

\[
\begin{aligned}
\text{minimize}\quad & \sum_{m\in M}\sum_{p\in P} C_{mp}\,x_{mp} \\[4pt]
\text{subject to}\quad 
& \sum_{p\in P} x_{mp}=1, &&\forall m\in M \\[2pt]
& \sum_{m\in M} x_{mp}=1, &&\forall p\in P \\[2pt]
& x_{mp}\in\{0,1\}, &&\forall m\in M,\;p\in P
\end{aligned}
\]

**Data Mapping**

| Parameter / Index | Source | Table ID | Key Column(s) | Value Column(s) |
|-------------------|--------|----------|---------------|-----------------|
| \(M\) (managers) | Index set | `file_0_view_0` | `Unnamed: 0` | – |
| \(P\) (projects) | Index set | `file_0_view_0` | – | `P1`, `P2`, `P3` |
| \(C_{mp}\) (cost) | Parameter | `file_0_view_0` | Row key: `Unnamed: 0` (manager), Column key: `P1`/`P2`/`P3` (project) | Cell value (numeric) |

All numeric values for \(C_{mp}\) are taken directly from the corresponding cells in the CSV. The binary decision variables \(x_{mp}\) are defined for every pair \((m,p)\) in the cross‑product of \(M\) and \(P\).