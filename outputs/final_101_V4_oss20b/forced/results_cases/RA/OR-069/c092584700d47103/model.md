**Mathematical Model**

\[
\begin{aligned}
\text{Minimize}\quad & \sum_{m\in M}\sum_{p\in P} c_{m,p}\,x_{m,p} \\[4pt]
\text{subject to}\quad
& \sum_{p\in P} x_{m,p}=1 && \forall m\in M \quad\text{(each manager assigned once)}\\
& \sum_{m\in M} x_{m,p}=1 && \forall p\in P \quad\text{(each project assigned once)}\\[4pt]
& x_{m,p}\in\{0,1\} && \forall m\in M,\;p\in P
\end{aligned}
\]

**Index Sets**

- \(M=\{\text{Manager 1},\dots,\text{Manager 11}\}\)
- \(P=\{\text{Project 1},\dots,\text{Project 11}\}\)

**Parameters**

- \(c_{m,p}\) – assignment cost for manager \(m\) to project \(p\).

**Decision Variables**

- \(x_{m,p}\) – binary variable, 1 if manager \(m\) is assigned to project \(p\), 0 otherwise.

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(c_{m,p}\) | `file_0_view_0` | For each \(m\in M\) and \(p\in P\): <br> `c_{m,p}` = value in column `"Project p Cost"` where `Manager = m`. |
| \(x_{m,p}\) | Decision variable | Binary variable defined in the model. |

*All column names are taken verbatim from the CSV: `"Project 1 Cost"`, `"Project 2 Cost"`, …, `"Project 11 Cost"`.*