## Mathematical Model

**Indices**  
\(W=\{1,2,3\}\) – workstations  
\(I=\{1,2,\dots ,101\}\) – radio models  

**Parameters**  
\(t_{w,i}\) – processing time (minutes) of model \(i\) on workstation \(w\)  
\(m_{w}\) – maintenance percent of workstation \(w\)  
\(C_{w}=1440\,(1-\frac{m_{w}}{100})\) – effective daily capacity of workstation \(w\)

**Decision Variables**  
\(x_{i}\in \mathbb{Z}_{+}\) – number of units of model \(i\) produced per day  

**Objective**  
\[
\min \;\sum_{w\in W}\Bigl(C_{w}-\sum_{i\in I}t_{w,i}\,x_{i}\Bigr)
\]
( total idle time of the three workstations )

**Constraints**  
\[
\sum_{i\in I}t_{w,i}\,x_{i}\;\le\;C_{w}\qquad \forall\,w\in W
\]
(used processing time cannot exceed effective capacity)

\[
x_{i}\;\ge\;0,\;\;x_{i}\in\mathbb{Z}\qquad \forall\,i\in I
\]

---

## Data Mapping

| Parameter | Source Table | Source Columns | Mapping Rule |
|-----------|--------------|----------------|--------------|
| \(t_{w,i}\) | `file_0_view_0` | `HiFi{i}_Minutes` | For each row where `Workstation = w`, read the value in column `HiFi{i}_Minutes`. |
| \(m_{w}\) | `file_0_view_0` | `Maintenance_Percent` | For each row where `Workstation = w`, read the value in column `Maintenance_Percent`. |
| \(C_{w}\) | Derived | – | Compute \(C_{w}=1440\,(1-\frac{m_{w}}{100})\) using the \(m_{w}\) values above. |

**Index Sets**

- \(W = \{1,2,3\}\) – obtained from distinct values of `Workstation` in the table.
- \(I = \{1,2,\dots ,101\}\) – fixed by the problem statement (HiFi‑1 … HiFi‑101).

All other data (processing times, maintenance percentages, and the constant 1440 minutes per day) are taken directly from the provided `workstation_times.csv` via the mapping rules above. No additional data or assumptions are introduced.