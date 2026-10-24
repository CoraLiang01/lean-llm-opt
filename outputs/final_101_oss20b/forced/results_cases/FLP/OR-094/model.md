**Mathematical Model**

Let  

* \(W=\{1,2,3\}\) – set of workstations  
* \(R=\{1,\dots ,101\}\) – set of radio models  

Parameters  

* \(t_{wr}\) – processing time (minutes) required for one unit of model \(r\) at workstation \(w\)  
* \(m_w\) – maintenance percentage of workstation \(w\) (e.g. 10, 14, 12)  
* \(C_w = 1440\,(1-m_w/100)\) – effective daily capacity of workstation \(w\) (minutes)  

Decision variables  

* \(x_{wr}\in \mathbb{Z}_{\ge 0}\) – number of units of model \(r\) produced at workstation \(w\)

Objective  

\[
\min \;\sum_{w\in W}\Bigl(C_w-\sum_{r\in R} t_{wr}\,x_{wr}\Bigr)
\]

(Equivalently, minimize total idle time across all workstations.)

Constraints  

1. **Capacity (idle‑time definition)** – already embedded in the objective; no additional constraints are required.  
2. **Non‑negativity & integrality**  

\[
x_{wr}\in \mathbb{Z}_{\ge 0}\quad \forall w\in W,\; r\in R
\]

---

**Data Mapping (from `workstation_times.csv`)**

| Symbol | Source column(s) | Source table | Notes |
|--------|------------------|--------------|-------|
| \(w\) | `Workstation` | `file_0_view_0` | Integer 1‑3 |
| \(r\) | `HiFi1_Minutes`, `HiFi2_Minutes`, … , `HiFi101_Minutes` | `file_0_view_0` | Each column corresponds to model \(r\) (e.g., `HiFi1_Minutes` → \(r=1\)) |
| \(t_{wr}\) | Value in row where `Workstation` = \(w\) and column `HiFi{r}_Minutes` | `file_0_view_0` | Processing time for model \(r\) at workstation \(w\) |
| \(m_w\) | `Maintenance_Percent` in row where `Workstation` = \(w\) | `file_0_view_0` | Maintenance percentage for workstation \(w\) |

The effective capacity \(C_w\) is computed in the model as \(1440(1-m_w/100)\). No other parameters are required.