**Mathematical Model**

Let  

* \(W=\{1,2,3\}\) – set of workstations  
* \(M=\{1,\dots ,101\}\) – set of radio models  

Parameters  

* \(t_{w,m}\) – processing time (minutes) required for one unit of model \(m\) at workstation \(w\)  
* \(p_{w}\) – maintenance percentage at workstation \(w\)  
* \(C_{w}=1440\,(1-p_{w}/100)\) – effective daily capacity of workstation \(w\)  

Decision variables  

* \(y_{m}\in \mathbb{Z}_{+}\) – number of units of model \(m\) produced per day  

Objective  

\[
\min \;\sum_{w\in W}\Bigl(C_{w}-\sum_{m\in M}t_{w,m}\,y_{m}\Bigr)
\]

(Equivalently, maximize \(\sum_{w}\sum_{m}t_{w,m}y_{m}\); the constant \(\sum_{w}C_{w}\) can be omitted.)

Constraints  

\[
\sum_{m\in M}t_{w,m}\,y_{m}\;\le\;C_{w}\qquad \forall\,w\in W
\]

\[
y_{m}\;\ge\;0,\;\; y_{m}\in \mathbb{Z}\qquad \forall\,m\in M
\]

---

**Data Mapping**

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(t_{w,m}\) | `file_0_view_0` | Column `HiFi{m}_Minutes` for row where `Workstation = w` |
| \(p_{w}\) | `file_0_view_0` | Column `Maintenance_Percent` for row where `Workstation = w` |
| \(C_{w}\) | Computed | \(1440\,(1-p_{w}/100)\) |

*Table ID:* `file_0_view_0`  
*Columns used:* `Workstation`, `HiFi1_Minutes` … `HiFi101_Minutes`, `Maintenance_Percent`