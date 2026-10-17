**Mathematical model**

Let  

* \(W=\{1,2,3\}\) – set of workstations  
* \(M=\{1,\dots ,101\}\) – set of radio models  

Parameters (obtained from *workstation_times.csv*)  

* \(t_{w,m}\) – processing time (minutes) required for one unit of model \(m\) on workstation \(w\)  
* \(p_w\) – maintenance percent for workstation \(w\)  
* \(C_w = 1440\,(1-p_w/100)\) – effective daily capacity of workstation \(w\) (minutes)

Decision variables  

* \(x_{w,m}\in \mathbb{Z}_{\ge 0}\) – number of units of model \(m\) produced on workstation \(w\)

---

\[
\begin{aligned}
\min\;& \sum_{w\in W}\Bigl(C_w-\sum_{m\in M} t_{w,m}\,x_{w,m}\Bigr) \\[4pt]
\text{s.t. } & \sum_{m\in M} t_{w,m}\,x_{w,m}\;\le\; C_w , &&\forall w\in W \\[4pt]
& x_{w,m}\;\in\;\mathbb{Z}_{\ge 0}, &&\forall w\in W,\; m\in M
\end{aligned}
\]

The objective is the total idle time across all workstations; the capacity constraints ensure that the processing time used on each workstation does not exceed its effective capacity.

---

**Data‑mapping rules**

| Parameter | Source column(s) | Mapping rule |
|-----------|------------------|--------------|
| \(t_{w,m}\) | `HiFi{m}_Minutes` | For each row with `Workstation = w`, read the value in column `HiFi{m}_Minutes` and assign to \(t_{w,m}\). |
| \(p_w\) | `Maintenance_Percent` | For each row with `Workstation = w`, read the value in column `Maintenance_Percent` and assign to \(p_w\). |
| \(C_w\) | – | Compute \(C_w = 1440\,(1-p_w/100)\) within the model (no explicit table entry). |

All indices \(w\) and \(m\) are taken from the rows and column names of the single table `file_0_view_0`. No other data are required.