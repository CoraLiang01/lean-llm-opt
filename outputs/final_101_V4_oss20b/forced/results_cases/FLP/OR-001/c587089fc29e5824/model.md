**Mathematical model**

Let  

* \(I=\{S1,\dots ,S18\}\) – set of distribution centers (suppliers)  
* \(J=\{C1,\dots ,C18\}\) – set of customer groups  

Parameters  

* \(d_j\) – daily demand of customer \(j\) (from *file_0_view_0*, column **demand**)  
* \(u_i\) – daily supply capacity of center \(i\) (from *file_1_view_0*, column **supply_capacity**)  
* \(c_{ij}\) – transportation cost per unit from center \(i\) to customer \(j\) (from *file_2_view_0*, columns **C1**–**C18**)  

Decision variables  

* \(x_{ij}\ge 0\) – quantity shipped from center \(i\) to customer \(j\)

\[
\begin{aligned}
\min\;&\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\;&\sum_{i\in I} x_{ij}=d_j , &&\forall j\in J \quad\text{(demand)}\\[4pt]
&\sum_{j\in J} x_{ij}\le u_i , &&\forall i\in I \quad\text{(capacity)}\\[4pt]
&x_{ij}\ge 0 , &&\forall i\in I,\;j\in J
\end{aligned}
\]

---

**Data‑mapping rules**

| Parameter | Source table | Index column | Value column |
|-----------|--------------|--------------|--------------|
| \(d_j\) | `file_0_view_0` | `customer` | `demand` |
| \(u_i\) | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` |
| \(c_{ij}\) | `file_2_view_0` | `Unnamed: 0` | `C1`–`C18` (column matching customer id) |

All index sets \(I,J\) are taken from the complete entity lists in the respective tables. No additional constraints or variables are introduced.