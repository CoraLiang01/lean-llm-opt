**Mathematical Model**

Let  

* \(I=\{1,\dots ,101\}\) – index of HiFi models  
* \(J=\{1,2,3\}\) – index of workstations  

Parameters  

* \(t_{ij}\) – processing time (minutes) of model \(i\) on workstation \(j\)  
* \(m_j\) – maintenance percent of workstation \(j\) (e.g. 10 for 10 %)  
* \(C_j = 1440\bigl(1-\frac{m_j}{100}\bigr)\) – effective daily capacity of workstation \(j\)

Decision variables  

* \(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of model \(i\) produced per day  

---

\[
\begin{aligned}
\min \quad & \sum_{j\in J}\Bigl(C_j-\sum_{i\in I} t_{ij}\,x_i\Bigr) \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} t_{ij}\,x_i \;\le\; C_j , \qquad \forall j\in J \\[4pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0}, \qquad \forall i\in I
\end{aligned}
\]

*(The objective is equivalent to maximizing the total used processing time, since \(\sum_j C_j\) is constant.)*

---

### Data Mapping

| Parameter | Source | Table ID | Column(s) |
|-----------|--------|----------|-----------|
| \(t_{ij}\) | Processing times | `file_0_view_0` | `HiFi{i}_Minutes` (for each \(i=1,\dots,101\)) |
| \(m_j\) | Maintenance percent | `file_0_view_0` | `Maintenance_Percent` (row‑wise, matched by `Workstation`) |
| \(C_j\) | Derived | – | \(1440\bigl(1-\frac{m_j}{100}\bigr)\) |
| Index \(J\) | Workstation IDs | `file_0_view_0` | `Workstation` |
| Index \(I\) | HiFi models | – | \(1,\dots,101\) (implicit from column names) |

All numeric values are taken directly from the CSV rows; no enumeration of individual rows is required in the model.