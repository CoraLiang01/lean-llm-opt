**Mathematical model**

Let  

* \(I=\{S1,S2,\dots ,S12\}\) – set of distribution centers (suppliers)  
* \(J=\{C1,C2,\dots ,C12\}\) – set of customer groups  

Parameters  

* \(d_j\) – daily demand of customer \(j\)  
* \(s_i\) – daily supply capacity of supplier \(i\)  
* \(c_{ij}\) – transportation cost per unit from supplier \(i\) to customer \(j\)  

Decision variables  

* \(x_{ij}\ge 0\) – quantity shipped from supplier \(i\) to customer \(j\)

\[
\begin{aligned}
\min\;&\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } &\sum_{i\in I} x_{ij} \;\ge\; d_j , &&\forall j\in J \quad\text{(demand)}\\[4pt]
&\sum_{j\in J} x_{ij} \;\le\; s_i , &&\forall i\in I \quad\text{(supply)}\\[4pt]
&x_{ij}\;\ge\;0 , &&\forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data mapping rules**

| Parameter | Source table | Key column | Value column(s) | Notes |
|-----------|--------------|------------|-----------------|-------|
| \(d_j\) | `file_0_view_0` | `customer` | `demand` | One row per customer |
| \(s_i\) | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` | One row per supplier |
| \(c_{ij}\) | `file_2_view_0` | `Unnamed: 0` | `C1` … `C12` | Matrix: rows = suppliers, columns = customers |

All indices \(i\) and \(j\) are taken from the respective key columns of the source tables. No additional aggregation or transformation is applied.