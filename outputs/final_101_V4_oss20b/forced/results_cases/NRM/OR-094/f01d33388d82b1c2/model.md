**Mathematical Model (symbolic)**  

Let  

* \(W=\{1,2,3\}\) – set of workstations  
* \(R=\{1,\dots ,101\}\) – set of radio models  

Parameters  

* \(t_{wr}\) – processing time (minutes) required for one unit of model \(r\) at workstation \(w\)  
* \(m_w\) – maintenance percent for workstation \(w\) (given in the data)  
* \(C_w = 1440\,(1-m_w/100)\) – effective daily capacity of workstation \(w\)  

Decision variables  

* \(x_r \in \mathbb{Z}_{\ge 0}\) – number of units of model \(r\) produced per day  

Objective (minimise total idle time)

\[
\min \; \sum_{w\in W}\Bigl(C_w-\sum_{r\in R} t_{wr}\,x_r\Bigr)
\]

Constraints  

1. **Capacity constraints** – for each workstation  
\[
\sum_{r\in R} t_{wr}\,x_r \;\le\; C_w , \qquad \forall w\in W
\]

2. **Non‑negativity and integrality** – for each model  
\[
x_r \;\in\; \mathbb{Z}_{\ge 0}, \qquad \forall r\in R
\]

---

### Data Mapping

| Symbol | Source | Table ID | Column(s) | Notes |
|--------|--------|----------|-----------|-------|
| \(t_{wr}\) | Processing times | `file_0_view_0` | `HiFi{r}_Minutes` (e.g., `HiFi1_Minutes`, `HiFi101_Minutes`) | One column per model, one row per workstation |
| \(m_w\) | Maintenance percent | `file_0_view_0` | `Maintenance_Percent` | Same column for all rows, value differs by workstation |
| \(C_w\) | Effective capacity | – | – | Computed as \(1440\,(1-m_w/100)\) |
| \(x_r\) | Decision variable | – | – | Not sourced from data |
| Objective | Idle time | – | – | Expression defined above |
| Constraints | Capacity & integrality | – | – | As listed above |

*All parameter values are taken directly from the three rows of `workstation_times.csv`.  
The mapping pattern for \(t_{wr}\) is: for each workstation row \(w\) and each model \(r\), the value in column `HiFi{r}_Minutes` is assigned to \(t_{wr}\).  
The maintenance percent \(m_w\) is taken from the `Maintenance_Percent` column of the same row.*