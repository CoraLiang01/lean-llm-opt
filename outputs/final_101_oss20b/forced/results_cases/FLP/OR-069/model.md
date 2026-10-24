**Mathematical Model**

Let  

* \(M=\{m_1,\dots ,m_{11}\}\) – set of managers (row identifiers)  
* \(P=\{p_1,\dots ,p_{11}\}\) – set of projects (column identifiers)  

Parameters  

* \(c_{ij}\) – assignment cost of manager \(m_i\) to project \(p_j\) (from the CSV).  

Decision variables  

* \(x_{ij}\in\{0,1\}\) – 1 if manager \(m_i\) is assigned to project \(p_j\), 0 otherwise.  

Objective  

\[
\min \;\sum_{i\in M}\sum_{j\in P} c_{ij}\,x_{ij}
\]

Constraints  

1. **Manager‑to‑project one‑to‑one**  
   \[
   \sum_{j\in P} x_{ij}=1 \quad \forall i\in M
   \]

2. **Project‑to‑manager one‑to‑one**  
   \[
   \sum_{i\in M} x_{ij}=1 \quad \forall j\in P
   \]

3. **Binary domain**  
   \[
   x_{ij}\in\{0,1\}\quad \forall i\in M,\; j\in P
   \]

---

**Data Mapping**

| Source | Table ID | Column Mapping |
|--------|----------|----------------|
| `manager_project_costs.csv` | `file_0_view_0` | `Manager` → manager index \(i\) <br> `Project k Cost` → project index \(j=k\) (for \(k=1,\dots,11\)) |

Parameter definition from the table:

\[
c_{ij} = \text{value in row where `Manager` = } m_i \text{ and column `Project } j \text{ Cost`}
\]

No other data are required. The model uses only the cost matrix; all other constraints are the standard one‑to‑one assignment constraints.