**Mathematical Model**

Let  

* \(W=\{1,2,3\}\) – set of workstations  
* \(M=\{1,\dots ,101\}\) – set of radio models  

Parameters (obtained from *workstation_times.csv*, table_id `file_0_view_0`)

| Symbol | Definition | Source mapping |
|--------|------------|----------------|
| \(t_{w,m}\) | Minutes required to process one unit of model \(m\) at workstation \(w\) | `file_0_view_0` – column `HiFi{m}_Minutes` in row where `Workstation = w` |
| \(p_w\) | Maintenance percent for workstation \(w\) | `file_0_view_0` – column `Maintenance_Percent` in row where `Workstation = w` |
| \(C_w\) | Effective daily capacity of workstation \(w\) | \(C_w = 1440\,(1-\frac{p_w}{100})\) |

Decision variables  

* \(x_m \in \mathbb{Z}_{\ge 0}\) – number of units of model \(m\) produced per day  

Objective (minimise total idle time)

\[
\min \;\sum_{w\in W}\Bigl(C_w-\sum_{m\in M} t_{w,m}\,x_m\Bigr)
\]

Capacity constraints (production may not exceed effective capacity)

\[
\sum_{m\in M} t_{w,m}\,x_m \;\le\; C_w , \qquad \forall\, w\in W
\]

Non‑negativity and integrality

\[
x_m \;\in\; \mathbb{Z}_{\ge 0}, \qquad \forall\, m\in M
\]

---

**Data Mapping Rules**

| Parameter | Table ID | Column(s) | Mapping rule |
|-----------|----------|-----------|--------------|
| \(t_{w,m}\) | `file_0_view_0` | `HiFi{m}_Minutes` | For each row where `Workstation = w`, read the value in column `HiFi{m}_Minutes`. |
| \(p_w\) | `file_0_view_0` | `Maintenance_Percent` | For each row where `Workstation = w`, read the value in column `Maintenance_Percent`. |
| \(C_w\) | – | – | Compute as \(1440\,(1-\frac{p_w}{100})\). |

All other data (e.g., the set of workstations \(W\) and models \(M\)) are derived from the structure of the table and the problem statement. No additional parameters or constraints are introduced.