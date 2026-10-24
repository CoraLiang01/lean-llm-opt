**Mathematical Model**

Let  

- \(M\) be the set of managers (rows of the CSV).  
- \(P\) be the set of projects (columns of the CSV, excluding the first column).  

Define the decision variable  

\[
x_{mp}\;=\;
\begin{cases}
1 & \text{if manager } m \text{ is assigned to project } p,\\
0 & \text{otherwise.}
\end{cases}
\]

Let \(c_{mp}\) denote the cost for manager \(m\) to complete project \(p\).

---

### Objective

\[
\min \;\sum_{m\in M}\sum_{p\in P} c_{mp}\,x_{mp}
\]

### Assignment Constraints

\[
\sum_{p\in P} x_{mp}=1 \quad \forall m\in M
\]

\[
\sum_{m\in M} x_{mp}=1 \quad \forall p\in P
\]

### Variable Domain

\[
x_{mp}\in\{0,1\}\quad \forall m\in M,\;p\in P
\]

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(c_{mp}\) | Table `file_0_view_0` | For each row where `Unnamed: 0 = m` and each column `Pj = p`, assign \(c_{mp} =\) the numeric value in that cell. |

*No numerical values are enumerated; the mapping references the CSV columns and rows directly.*