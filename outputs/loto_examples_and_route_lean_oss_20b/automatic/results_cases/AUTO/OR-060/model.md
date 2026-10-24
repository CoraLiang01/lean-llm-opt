**Mathematical Model**

Let  

* \(I=\{S_1,\dots ,S_{12}\}\) – suppliers  
* \(J=\{C_1,\dots ,C_{12}\}\) – supermarkets  

Parameters  

* \(f_i\) – fixed opening cost of supplier \(i\)  
* \(d_j\) – demand of supermarket \(j\) (units)  
* \(c_{ij}\) – transportation cost per unit from supplier \(i\) to supermarket \(j\)

Decision variables  

* \(y_i\in\{0,1\}\) – 1 if supplier \(i\) is opened, 0 otherwise  
* \(x_{ij}\ge 0\) – quantity supplied from \(i\) to \(j\)

\[
\begin{aligned}
\min\;&\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } &\sum_{i\in I} x_{ij}=d_j \qquad &&\forall j\in J \quad\text{(demand)}\\[4pt]
&x_{ij}\;\le\; d_j\,y_i \qquad &&\forall i\in I,\;j\in J \quad\text{(supplier open)}\\[4pt]
&y_i\in\{0,1\} \qquad &&\forall i\in I\\
&x_{ij}\ge 0 \qquad &&\forall i\in I,\;j\in J
\end{aligned}
\]

---

### Data Mapping

| Parameter | Source Table | Row Identifier | Column Identifier | Notes |
|-----------|--------------|----------------|-------------------|-------|
| \(f_i\) | `file_1_view_0` | `Unnamed: 0` (supplier name) | `fixed_costs` | Fixed cost per supplier |
| \(d_j\) | `file_0_view_0` | `customer` | `demand` | Demand per supermarket |
| \(c_{ij}\) | `file_2_view_0` | `Unnamed: 0` (supplier name) | `C1`–`C12` (customer columns) | Transportation cost per unit |

*Index sets*  
- \(I\) is the set of supplier identifiers extracted from the `Unnamed: 0` column of `file_1_view_0`.  
- \(J\) is the set of customer identifiers extracted from the `customer` column of `file_0_view_0`.

All parameters are taken directly from the corresponding tables; no additional numerical processing or enumeration of values is performed in the model.