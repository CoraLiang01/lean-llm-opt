**Mathematical Model**

\[
\begin{aligned}
\text{Minimize}\quad
&\sum_{j\in J}\Bigl(C_j-\sum_{i\in I}t_{j,i}\,x_i\Bigr) \\[4pt]
\text{subject to}\quad
&\sum_{i\in I}t_{j,i}\,x_i \;\le\; C_j , &&\forall j\in J \\[4pt]
&x_i \;\in\;\mathbb{Z}_{\ge 0}, &&\forall i\in I
\end{aligned}
\]

where  

* \(I=\{1,\dots ,101\}\) – set of HiFi models  
* \(J=\{1,2,3\}\) – set of workstations  
* \(t_{j,i}\) – processing time (minutes) of model \(i\) on workstation \(j\)  
* \(m_j\) – maintenance percent of workstation \(j\)  
* \(C_j = 1440\bigl(1-\frac{m_j}{100}\bigr)\) – effective daily capacity of workstation \(j\)  
* \(x_i\) – number of units of model \(i\) produced per day  

---

**Data Mapping**

| Parameter | Source File | Source Columns | Notes |
|-----------|-------------|----------------|-------|
| \(t_{j,i}\) | `Test_Dataset/Large-scale-or/Mixture_testing/Mixture13/workstation_times.csv` | `HiFi{i}_Minutes` (columns 2–102) | Row where `Workstation = j` |
| \(m_j\) | same file | `Maintenance_Percent` (column 103) | Row where `Workstation = j` |
| \(C_j\) | derived | – | \(C_j = 1440(1-m_j/100)\) |

All other parameters are constants derived from the above data. No additional data are required.