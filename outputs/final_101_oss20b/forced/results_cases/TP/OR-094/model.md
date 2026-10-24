**Mathematical model**

Let  

* \(W=\{1,2,3\}\) – set of workstations  
* \(M=\{1,\dots ,101\}\) – set of radio models  

Parameters  

* \(t_{w,m}\) – processing time (minutes) required for one unit of model \(m\) on workstation \(w\)  
* \(p_{w}\) – maintenance percentage for workstation \(w\)  
* \(C_{w}=1440\,(1-p_{w}/100)\) – effective daily capacity of workstation \(w\) (minutes)  

Decision variables  

* \(x_{w,m}\in \mathbb{Z}_{+}\) – number of units of model \(m\) produced on workstation \(w\)

---

\[
\begin{aligned}
\min\;&\sum_{w\in W}\Bigl(C_{w}-\sum_{m\in M}t_{w,m}\,x_{w,m}\Bigr)\\[4pt]
\text{s.t. }&\sum_{m\in M}t_{w,m}\,x_{w,m}\;\le\;C_{w},
\qquad \forall\,w\in W\\[4pt]
&x_{w,m}\;\in\;\mathbb{Z}_{+},
\qquad \forall\,w\in W,\;m\in M
\end{aligned}
\]

The objective is the total idle time of the three workstations; the capacity
constraints guarantee that the processing time used on each workstation does
not exceed its effective capacity.

---

**Data‑mapping rules**

| Source column | Parameter | Mapping rule |
|---------------|-----------|--------------|
| `Workstation` | \(w\) | integer value 1, 2, or 3 |
| `HiFi1_Minutes` … `HiFi101_Minutes` | \(t_{w,m}\) | for each column `HiFiX_Minutes`, set \(m=X\) and assign the numeric value to \(t_{w,m}\) |
| `Maintenance_Percent` | \(p_{w}\) | assign the numeric value to \(p_{w}\) |

All values are taken directly from the rows of **file_0_view_0** (the
`workstation_times.csv` view).  No additional data or constraints are
introduced beyond those specified in the query.