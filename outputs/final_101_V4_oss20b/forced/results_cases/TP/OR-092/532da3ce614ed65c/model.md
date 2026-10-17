**Mathematical Model**

Let  

* \(I=\{S1,\dots ,S10\}\) – source locations  
* \(J=\{D1,\dots ,D20\}\) – destination locations  

Parameters (all read from the supplied CSV files)

* \(c_{ij}\) – unit transportation cost from source \(i\) to destination \(j\)  
  (file **expanded_cost_matrix.csv**, row = source_id, column = destination_id)  
* \(d_j\) – demand units at destination \(j\)  
  (file **expanded_destinations.csv**, column = demand_units)  
* \(s_i\) – supply units available at source \(i\)  
  (file **expanded_sources.csv**, column = supply_units)  
* \(Q=10\) – maximum cargo units that can be loaded on one truck  

Decision variables  

* \(x_{ij}\ge 0\) – cargo units shipped from \(i\) to \(j\) (continuous)  
* \(t_{ij}\in \mathbb{Z}_{+}\) – number of whole trucks dispatched from \(i\) to \(j\)

\[
\begin{aligned}
\min\;&\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } &\sum_{i\in I} x_{ij}\;\ge\; d_j , &&\forall j\in J \qquad\text{(demand)}\\[4pt]
&\sum_{j\in J} x_{ij}\;\le\; s_i , &&\forall i\in I \qquad\text{(supply)}\\[4pt]
&x_{ij}\;\le\; Q\,t_{ij} , &&\forall i\in I,\; j\in J \qquad\text{(truck capacity)}\\[4pt]
&x_{ij}\;\ge\;0 , &&\forall i\in I,\; j\in J \\[2pt]
&t_{ij}\;\in\mathbb{Z}_{+}, &&\forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source File | Row Key | Column Key | Notes |
|-----------|-------------|---------|------------|-------|
| \(c_{ij}\) | `expanded_cost_matrix.csv` | `source_id` | `destination_id` | 10 × 20 matrix |
| \(d_j\) | `expanded_destinations.csv` | `destination_id` | `demand_units` | 20 rows |
| \(s_i\) | `expanded_sources.csv` | `source_id` | `supply_units` | 10 rows |

All identifiers (`S1`–`S10`, `D1`–`D20`) are taken verbatim from the CSV headers and rows. No aggregation or renaming is performed. The truck capacity \(Q\) is a fixed constant (10 units) as specified in the problem statement.