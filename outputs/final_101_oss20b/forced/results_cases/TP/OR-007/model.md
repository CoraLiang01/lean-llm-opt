**Mathematical Model**

Let  

- \(I=\{i\mid i\in\text{file\_1\_view\_0.region}\}\) (warehouses)  
- \(J=\{j\mid j\in\text{file\_0\_view\_0.customer}\}\) (store locations)

Parameters  

- \(d_j\) = demand of store \(j\) (from **file_0_view_0**.demand)  
- \(s_i\) = supply capacity of warehouse \(i\) (from **file_1_view_0**.supply_capacity)  
- \(c_{ij}\) = transportation cost per unit from warehouse \(i\) to store \(j\) (from **file_2_view_0**; row id = \(i\), column id = \(j\))

Decision variables  

- \(x_{ij}\ge 0\) = quantity shipped from warehouse \(i\) to store \(j\)

\[
\begin{aligned}
\min\;&\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\;&\sum_{i\in I} x_{ij}\;\ge\; d_j , &&\forall j\in J \quad\text{(demand)}\\
&\sum_{j\in J} x_{ij}\;\le\; s_i , &&\forall i\in I \quad\text{(supply)}\\
&x_{ij}\;\ge\;0 , &&\forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Source Column | Mapping |
|-----------|--------------|---------------|---------|
| \(I\) | **file_1_view_0** | region | set of all distinct region values |
| \(J\) | **file_0_view_0** | customer | set of all distinct customer values |
| \(d_j\) | **file_0_view_0** | demand | value where customer = \(j\) |
| \(s_i\) | **file_1_view_0** | supply_capacity | value where region = \(i\) |
| \(c_{ij}\) | **file_2_view_0** | D1–D5 columns | value at row where Unnamed: 0 = \(i\) and column = \(j\) |

All indices, parameters, and variables are defined exactly as above; no additional data or constraints are introduced.