## Mathematical Model

**Indices**  
\(W=\{1,2,3\}\) – workstations  
\(I=\{1,2,\dots ,101\}\) – radio models  

**Parameters**  
\(t_{w,i}\) – processing time (minutes) of model \(i\) on workstation \(w\)  
\(m_w\) – maintenance percent of workstation \(w\) (as a fraction)  
\(C_w = 1440\,(1-m_w)\) – effective daily capacity of workstation \(w\) (minutes)

**Decision variables**  
\(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of model \(i\) produced per day  

**Objective** – minimize total idle time  
\[
\min \;\sum_{w\in W}\Bigl(C_w-\sum_{i\in I} t_{w,i}\,x_i\Bigr)
\]

**Constraints** – capacity limits  
\[
\sum_{i\in I} t_{w,i}\,x_i \;\le\; C_w \qquad \forall\,w\in W
\]

**Variable domains**  
\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall\,i\in I
\]

---

## Data Mapping

| Table ID | Column | Mapping |
|----------|--------|---------|
| `file_0_view_0` | `Workstation` | \(w\) |
| `file_0_view_0` | `Maintenance_Percent` | \(m_w = \text{Maintenance_Percent}/100\) |
| `file_0_view_0` | `HiFi1_Minutes` … `HiFi101_Minutes` | \(t_{w,i}\) for \(i=1,\dots ,101\) |

*The effective capacity \(C_w\) is computed from the mapped \(m_w\) as \(C_w = 1440(1-m_w)\). All other parameters are taken directly from the corresponding columns.*