**Mathematical model**

Sets  
\[
I=\{1,\dots ,101\}\qquad\text{(HiFi models)}\\
J=\{1,2,3\}\qquad\text{(workstations)}
\]

Parameters  
\[
t_{ij}\ge 0 \quad\text{processing time (min) of model }i\text{ on workstation }j\\
M_j\in[0,1] \quad\text{maintenance fraction of workstation }j\\
C_j=1440\,(1-M_j)\quad\text{effective capacity (min) of workstation }j
\]

Decision variables  
\[
x_i\in\mathbb{Z}_{\ge 0}\qquad\text{units of model }i\text{ produced per day}
\]

Objective (minimise total idle time)  
\[
\min \;\sum_{j\in J}\Bigl(C_j-\sum_{i\in I}t_{ij}\,x_i\Bigr)
\]

Constraints  
\[
\sum_{i\in I}t_{ij}\,x_i \;\le\; C_j,\qquad \forall j\in J
\]

All variables \(x_i\) are non‑negative integers.

---

**Data‑mapping rules**

| Parameter / Set | Source table | Source column(s) | Mapping rule |
|-----------------|--------------|------------------|--------------|
| \(I\) | `file_0_view_0` | columns `HiFi1_Minutes` … `HiFi101_Minutes` | \(I=\{1,\dots ,101\}\) (each column name defines an index \(i\)) |
| \(J\) | `file_0_view_0` | column `Workstation` | \(J=\{\text{distinct values of Workstation}\}\) |
| \(t_{ij}\) | `file_0_view_0` | column `HiFi{i}_Minutes` | For each row with `Workstation=j`, read the value in column `HiFi{i}_Minutes` |
| \(M_j\) | `file_0_view_0` | column `Maintenance_Percent` | For each row with `Workstation=j`, set \(M_j=\text{Maintenance_Percent}_j/100\) |
| \(C_j\) | – | – | Compute \(C_j=1440\,(1-M_j)\) from the mapped \(M_j\) |

No other data are required. The model uses only the parameters and sets defined above.