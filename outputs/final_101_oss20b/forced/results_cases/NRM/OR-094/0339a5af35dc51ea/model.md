##### Index Sets  

\[
I=\{1,2,\dots ,101\}\quad\text{(radio models HiFi-1 to HiFi-101)}  
\]  

\[
J=\{1,2,3\}\quad\text{(workstations 1, 2, 3)}  
\]  

##### Parameters  

\[
t_{ij}\quad\text{processing time (minutes) of model }i\text{ at workstation }j  
\]  

\[
C_{j}=1440\Bigl(1-\frac{m_{j}}{100}\Bigr)\quad\text{effective daily capacity of workstation }j  
\]  

where  

\[
m_{j}\quad\text{maintenance percent of workstation }j  
\]  

##### Variables  

\[
x_{i}\in \mathbb{Z}_{\ge 0}\quad\text{number of units of model }i\text{ produced per day}  
\]  

##### Objective Function  

\[
\min \;\sum_{j\in J}\Bigl(C_{j}-\sum_{i\in I}t_{ij}\,x_{i}\Bigr)
\]  

##### Constraints  

###### 1. Workstation capacity constraints  

\[
\sum_{i\in I}t_{ij}\,x_{i}\;\le\;C_{j}\quad\forall j\in J
\]  

###### 2. Non‑negativity and integrality  

\[
x_{i}\in \mathbb{Z}_{\ge 0}\quad\forall i\in I
\]  

##### Data Mapping  

| Parameter | Source table_id | Source column(s) | Mapping description |
|-----------|-----------------|------------------|---------------------|
| \(t_{ij}\) | `file_0_view_0` | `HiFi{i}_Minutes` | For each workstation row \(j\), the column `HiFi{i}_Minutes` gives the processing time of model \(i\). |
| \(m_{j}\) | `file_0_view_0` | `Maintenance_Percent` | The column `Maintenance_Percent` in row \(j\) gives the maintenance percentage for workstation \(j\). |
| \(C_{j}\) | `file_0_view_0` | `Workstation`, `Maintenance_Percent` | Compute \(C_{j}=1440(1-m_{j}/100)\) using the maintenance percent from the row where `Workstation` = \(j\). |

All other data (e.g., the list of models, the number of workstations, the total daily time 1440 minutes) are constants defined in the model description.