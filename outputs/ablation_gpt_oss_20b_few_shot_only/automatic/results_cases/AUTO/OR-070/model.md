**Mathematical Model**

Let  

- \(M=\{m_1,\dots ,m_7\}\) be the set of managers (Manager 1 … Manager 7).  
- \(P=\{p_1,\dots ,p_7\}\) be the set of projects (Project 1 … Project 7).  

Parameters  

- \(c_{mp}\) – cost for manager \(m\) to complete project \(p\).  

Decision variables  

- \(x_{mp}\in\{0,1\}\) – equals 1 if manager \(m\) is assigned to project \(p\), 0 otherwise.  

**Objective**  

\[
\min \sum_{m\in M}\sum_{p\in P} c_{mp}\,x_{mp}
\]

**Constraints**

1. **Each manager is assigned to exactly one project**

\[
\sum_{p\in P} x_{mp}=1 \qquad \forall m\in M
\]

2. **Each project is assigned to exactly one manager**

\[
\sum_{m\in M} x_{mp}=1 \qquad \forall p\in P
\]

3. **Binary variables**

\[
x_{mp}\in\{0,1\}\qquad \forall m\in M,\;p\in P
\]

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(c_{mp}\) | `Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv` | \(c_{m_i,p_j}\) = value in row *i* (Manager i) and column *j* (Project j Cost) of the CSV file. |

The index sets \(M\) and \(P\) are derived from the row and column headers of the CSV file, respectively. No numerical values are enumerated in the model; all coefficients are linked to the source data via the mapping rule above.