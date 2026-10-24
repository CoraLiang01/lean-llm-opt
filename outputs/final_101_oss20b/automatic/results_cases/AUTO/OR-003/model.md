**Mathematical Model**

Let  

- \(I=\{S1,S2,\dots ,S10\}\) ( suppliers )  
- \(J=\{C1,C2,\dots ,C10\}\) ( customer groups )  

Parameters  

- \(d_j\) : daily demand of customer \(j\)  \(j\in J\)  
- \(s_i\) : daily supply capacity of supplier \(i\)  \(i\in I\)  
- \(c_{ij}\) : transportation cost per unit from supplier \(i\) to customer \(j\) \(i\in I,\;j\in J\)  

Decision variables  

- \(x_{ij}\ge 0\) : quantity shipped from supplier \(i\) to customer \(j\) \(i\in I,\;j\in J\)

\[
\begin{aligned}
\min\;&\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } &\sum_{i\in I} x_{ij} \;\ge\; d_j , &&\forall j\in J \quad\text{(demand)}\\[4pt]
&\sum_{j\in J} x_{ij} \;\le\; s_i , &&\forall i\in I \quad\text{(supply)}\\[4pt]
&x_{ij}\;\ge\;0 , &&\forall i\in I,\;j\in J
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Source Columns | Mapping |
|-----------|--------------|----------------|---------|
| \(d_j\) | `file_0_view_0` | `customer`, `demand` | `d_j = demand` where `customer = j` |
| \(s_i\) | `file_1_view_0` | `Unnamed: 0`, `supply_capacity` | `s_i = supply_capacity` where `Unnamed: 0 = i` |
| \(c_{ij}\) | `file_2_view_0` | `Unnamed: 0`, `C1`–`C10` | `c_{ij} = value` where `Unnamed: 0 = i` and column `Cj` = j |

All indices, parameters, and variables are defined exactly as above; no additional data or constraints are introduced.