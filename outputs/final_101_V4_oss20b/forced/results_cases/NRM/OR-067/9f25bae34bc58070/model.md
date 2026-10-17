**Mathematical Model**

\[
\begin{aligned}
\text{Sets} \quad & M \quad\text{: managers} \\
& P \quad\text{: projects} \\[4pt]

\text{Parameters} \quad & c_{m,p} \quad\text{: cost of assigning manager }m\text{ to project }p \\[4pt]

\text{Decision Variables} \quad & x_{m,p}\in\{0,1\}\quad\forall m\in M,\;p\in P \\[4pt]

\text{Objective} \quad & \min Z = \sum_{m\in M}\sum_{p\in P} c_{m,p}\,x_{m,p} \\[6pt]

\text{Constraints} \\
& \sum_{p\in P} x_{m,p}=1 \qquad \forall m\in M \quad\text{(each manager gets one project)}\\
& \sum_{m\in M} x_{m,p}=1 \qquad \forall p\in P \quad\text{(each project gets one manager)}\\
\end{aligned}
\]

---

**Data Mapping**

| Parameter / Set | Source Table | Source Columns | Mapping Rule |
|-----------------|--------------|----------------|--------------|
| \(M\) | `file_0_view_0` | `Unnamed: 0` | Each distinct value in column `Unnamed: 0` defines a manager \(m\). |
| \(P\) | `file_0_view_0` | `P1`, `P2`, `P3` | Each column name `P1`, `P2`, `P3` defines a project \(p\). |
| \(c_{m,p}\) | `file_0_view_0` | `P1`, `P2`, `P3` | For a given manager \(m\) (row where `Unnamed: 0` = \(m\)) and project \(p\) (column `Pp`), the cost value in that cell is \(c_{m,p}\). |
| \(x_{m,p}\) | — | — | Binary decision variable, domain \(\{0,1\}\). |

*All numerical values are taken directly from the CSV; no values are hard‑coded in the model.*