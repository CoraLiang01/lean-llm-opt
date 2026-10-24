**Mathematical Model**

Let  

* \(W=\{1,2,3\}\) – set of workstations.  
* \(R=\{1,\dots ,101\}\) – set of radio models.  

Decision variables  

\[
x_{r,w}\in \mathbb{Z}_{\ge 0}\qquad \forall r\in R,\; w\in W
\]

Parameters  

\[
t_{r,w}\;(\text{minutes per unit})\qquad \forall r\in R,\; w\in W
\]
\[
m_{w}\;(\text{maintenance percent})\qquad \forall w\in W
\]
\[
C_{w}=1440\Bigl(1-\frac{m_{w}}{100}\Bigr)\qquad \forall w\in W
\]

Objective (minimise total idle time)

\[
\min \;\sum_{w\in W}\Bigl(C_{w}-\sum_{r\in R}t_{r,w}\,x_{r,w}\Bigr)
\]

Capacity constraints (effective capacity of each workstation)

\[
\sum_{r\in R}t_{r,w}\,x_{r,w}\;\le\;C_{w}\qquad \forall w\in W
\]

All variables are non‑negative integers.

---

### Data Mapping

| Parameter | Mapping Rule (table_id, column) |
|-----------|---------------------------------|
| \(t_{r,w}\) | `file_0_view_0[Workstation = w, column = HiFi{r}_Minutes]` |
| \(m_{w}\) | `file_0_view_0[Workstation = w, column = Maintenance_Percent]` |
| \(C_{w}\) | `1440 * (1 - m_{w}/100)` (computed from \(m_{w}\)) |

*Index sets*  
- \(W\) is obtained from the distinct values of the `Workstation` column.  
- \(R\) is the set of integers 1 through 101, corresponding to the columns `HiFi1_Minutes` … `HiFi101_Minutes`.  

All other data (e.g., processing times, maintenance percentages) are taken directly from the rows of `file_0_view_0`.