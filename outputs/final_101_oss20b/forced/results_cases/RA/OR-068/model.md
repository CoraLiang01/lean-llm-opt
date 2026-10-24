**Mathematical Model**

\[
\begin{aligned}
\text{minimize}\quad & \sum_{m\in M}\sum_{p\in P} c_{mp}\,x_{mp} \\[4pt]
\text{subject to}\quad
& \sum_{p\in P} x_{mp}=1 && \forall m\in M \quad\text{(each manager gets one project)}\\
& \sum_{m\in M} x_{mp}=1 && \forall p\in P \quad\text{(each project gets one manager)}\\[4pt]
& x_{mp}\in\{0,1\} && \forall m\in M,\;p\in P
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Mapping Rule |
|--------|--------|--------------|
| \(M\) | `file_0_view_0` | Set of all values in column **`Unnamed: 0`** (manager identifiers) |
| \(P\) | `file_0_view_0` | Set of all column names **`P1`** … **`P6`** (project identifiers) |
| \(c_{mp}\) | `file_0_view_0` | For each manager \(m\) (row where `Unnamed: 0` = \(m\)) and each project \(p\) (column `Pp`), the cost value in that cell is \(c_{mp}\) |

All other data (e.g., variable domains) are defined by the model itself. No numerical values are hard‑coded; they are retrieved from the specified table and columns.