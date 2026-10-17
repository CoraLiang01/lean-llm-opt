**Mathematical Model**

Let  

- \(M\) be the set of managers,  
- \(P\) be the set of projects,  
- \(c_{mp}\) be the cost for manager \(m\in M\) to complete project \(p\in P\),  
- \(x_{mp}\in\{0,1\}\) be a binary decision variable that equals 1 if manager \(m\) is assigned to project \(p\).

\[
\begin{aligned}
\min\;& \sum_{m\in M}\sum_{p\in P} c_{mp}\,x_{mp} \\[4pt]
\text{s.t. } & \sum_{p\in P} x_{mp}=1 && \forall m\in M \quad\text{(each manager gets one project)}\\
& \sum_{m\in M} x_{mp}=1 && \forall p\in P \quad\text{(each project gets one manager)}\\
& x_{mp}\in\{0,1\} && \forall m\in M,\;p\in P
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Mapping Rule |
|--------|--------|--------------|
| \(M\) | `file_0_view_0` | Set of distinct values in column **`Unnamed: 0`** (manager identifiers). |
| \(P\) | `file_0_view_0` | Set of column names **`P1`, `P2`, `P3`** (project identifiers). |
| \(c_{mp}\) | `file_0_view_0` | For each row where `Unnamed: 0 = m` and each column `p`, assign the numeric value in that cell to \(c_{mp}\). |
| \(x_{mp}\) | Decision variable | Binary variable defined for every pair \((m,p)\) with \(m\in M,\;p\in P\). |

*All numeric values are taken directly from the CSV; no enumeration or simplification is performed.*