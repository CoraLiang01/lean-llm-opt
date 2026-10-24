**Mathematical model**

Let  

* \(I=\{S1,S2,\dots ,S18\}\) – set of distribution centers (suppliers)  
* \(J=\{C1,C2,\dots ,C18\}\) – set of customer groups  

Parameters  

* \(d_j\) – daily demand of customer \(j\)  (from *file_0_view_0*, columns *customer*, *demand*)  
* \(s_i\) – daily supply capacity of supplier \(i\)  (from *file_1_view_0*, columns *Unnamed: 0*, *supply_capacity*)  
* \(c_{ij}\) – transportation cost per unit from supplier \(i\) to customer \(j\)  (from *file_2_view_0*, row id *Unnamed: 0* = \(i\), column id = \(j\))

Decision variables  

\[
x_{ij}\;\ge 0 \qquad \forall i\in I,\; j\in J
\]

Objective  

\[
\min \;\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

Constraints  

1. **Demand satisfaction**  
\[
\sum_{i\in I} x_{ij} \;\ge\; d_j \qquad \forall j\in J
\]

2. **Supply capacity**  
\[
\sum_{j\in J} x_{ij} \;\le\; s_i \qquad \forall i\in I
\]

3. **Non‑negativity** (already in variable definition)  

---

**Data mapping rules**

| Parameter | Source table | Row key | Column key | Notes |
|-----------|--------------|---------|------------|-------|
| \(d_j\) | `file_0_view_0` | `customer` | `demand` | One row per customer |
| \(s_i\) | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` | One row per supplier |
| \(c_{ij}\) | `file_2_view_0` | `Unnamed: 0` (row) | `C1`–`C18` (columns) | 18×18 cost matrix |

These mappings provide all numerical values needed to instantiate the model. No additional data or constraints are required.