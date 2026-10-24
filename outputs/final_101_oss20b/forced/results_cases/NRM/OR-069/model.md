**Mathematical Model**

\[
\begin{aligned}
\text{Sets} \quad & M \;=\;\{\text{Manager 1},\dots,\text{Manager 11}\},\\
& P \;=\;\{1,2,\dots,11\}.\\[4pt]

\text{Parameters} \quad & c_{m,p}\;\; (m\in M,\;p\in P) \quad\text{: assignment cost}.\\[4pt]

\text{Decision Variables} \quad & x_{m,p}\;\in\{0,1\}\quad(m\in M,\;p\in P).\\[4pt]

\text{Objective} \quad & \min \;\sum_{m\in M}\sum_{p\in P} c_{m,p}\,x_{m,p}.\\[6pt]

\text{Constraints} \\
&\sum_{p\in P} x_{m,p}=1 \qquad \forall m\in M \quad\text{(each manager assigned to one project)}\\[4pt]
&\sum_{m\in M} x_{m,p}=1 \qquad \forall p\in P \quad\text{(each project assigned to one manager)}\\[4pt]
&x_{m,p}\in\{0,1\}\qquad \forall m\in M,\;p\in P.
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(c_{m,p}\) | `file_0_view_0` | For each manager \(m\) (row where **Manager** = \(m\)) and each project \(p\) (column **Project \(p\) Cost**), assign the numeric value in that cell to \(c_{m,p}\). |
| Sets \(M\) | `file_0_view_0` | All distinct values in the **Manager** column. |
| Sets \(P\) | `file_0_view_0` | The project indices 1–11 corresponding to columns **Project 1 Cost** through **Project 11 Cost**. |

These rules bind every parameter and index set directly to the CSV source without introducing any additional data.