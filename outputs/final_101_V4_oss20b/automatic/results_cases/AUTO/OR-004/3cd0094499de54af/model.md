**Mathematical model**

Let  

* \(I=\{S1,S2,\dots ,S12\}\) – set of distribution centers (suppliers).  
* \(J=\{C1,C2,\dots ,C12\}\) – set of customer groups.  

Parameters  

\[
\begin{aligned}
d_j &\;=\;\text{daily demand of customer }j &&\text{(from table }file_0_view_0,\text{ column }demand)\\
s_i &\;=\;\text{daily supply capacity of supplier }i &&\text{(from table }file_1_view_0,\text{ column }supply_capacity)\\
c_{ij}&\;=\;\text{transportation cost per unit from supplier }i\text{ to customer }j &&\text{(from table }file_2_view_0,\text{ columns }C1\!-\!C12)
\end{aligned}
\]

Decision variables  

\[
x_{ij}\;\ge 0 \qquad \forall\, i\in I,\; j\in J
\]

Objective  

\[
\min \;\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

Constraints  

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}\;\ge\; d_j \qquad \forall\, j\in J
   \]

2. **Supply capacity**  
   \[
   \sum_{j\in J} x_{ij}\;\le\; s_i \qquad \forall\, i\in I
   \]

3. **Non‑negativity** (already stated in variable definition).  

---

**Data mapping**

| Parameter | Table ID | Column(s) | Notes |
|-----------|----------|-----------|-------|
| \(d_j\) | `file_0_view_0` | `demand` | Row key: `customer` |
| \(s_i\) | `file_1_view_0` | `supply_capacity` | Row key: `Unnamed: 0` |
| \(c_{ij}\) | `file_2_view_0` | `C1`–`C12` | Row key: `Unnamed: 0` (supplier) |

All indices \(i\) and \(j\) are taken from the row and column identifiers of the respective tables, ensuring that the model uses exactly the entities present in the current data. No additional aggregation or transformation is performed.