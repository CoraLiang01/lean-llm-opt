**Mathematical Model**

Let  

* \(W=\{1,2,3\}\) – set of workstations  
* \(R=\{1,\dots ,101\}\) – set of radio models  

Parameters  

* \(t_{wr}\) – processing time (minutes) required for one unit of model \(r\) at workstation \(w\)  
* \(m_w\) – maintenance percentage for workstation \(w\)  
* \(C_w = 1440\,(1-m_w/100)\) – effective daily capacity of workstation \(w\) (minutes)

Decision variables  

* \(x_r \in \mathbb{Z}_{\ge 0}\) – number of units of model \(r\) produced per day  

Objective (minimise total idle time)

\[
\min \;\; \sum_{w\in W}\Bigl(C_w-\sum_{r\in R} t_{wr}\,x_r\Bigr)
\]

Constraints  

1. **Capacity limits** – production at each workstation cannot exceed its effective capacity  

\[
\sum_{r\in R} t_{wr}\,x_r \;\le\; C_w , \qquad \forall w\in W
\]

2. **Non‑negativity and integrality**  

\[
x_r \in \mathbb{Z}_{\ge 0}, \qquad \forall r\in R
\]

---

**Data Mapping**

| Parameter | Source table | Source columns | Mapping rule |
|-----------|--------------|----------------|--------------|
| \(t_{wr}\) | `file_0_view_0` | `HiFi{r}_Minutes` (e.g., `HiFi1_Minutes`, `HiFi2_Minutes`, …, `HiFi101_Minutes`) | For each workstation row \(w\), read the value in column `HiFi{r}_Minutes` and assign to \(t_{wr}\). |
| \(m_w\) | `file_0_view_0` | `Maintenance_Percent` | For each workstation row \(w\), read the value in column `Maintenance_Percent` and assign to \(m_w\). |
| \(C_w\) | Derived | – | Compute \(C_w = 1440\,(1-m_w/100)\) for each workstation \(w\). |

All other data (e.g., total capacity, number of models) are constants derived from the problem statement and need not be stored separately.