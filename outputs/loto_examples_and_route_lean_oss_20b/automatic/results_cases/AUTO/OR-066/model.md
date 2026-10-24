**Mathematical Model**

Let  

* \(I=\{S1,S2\}\) – set of suppliers  
* \(J=\{C1,C2\}\) – set of supermarkets  

Parameters  

* \(f_i\) – fixed activation cost of supplier \(i\)  
* \(t_{ij}\) – transportation cost per unit from supplier \(i\) to supermarket \(j\)  
* \(d_j\) – demand of supermarket \(j\)  

Decision variables  

* \(y_i\in\{0,1\}\) – 1 if supplier \(i\) is activated, 0 otherwise  
* \(x_{ij}\ge 0\) – quantity supplied from supplier \(i\) to supermarket \(j\)

\[
\begin{aligned}
\min\;&\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} t_{ij}\,x_{ij} \\[4pt]
\text{s.t. }&\sum_{i\in I} x_{ij}=d_j \qquad &&\forall j\in J \quad\text{(demand)}\\[4pt]
&x_{ij}\le d_j\,y_i \qquad &&\forall i\in I,\; j\in J \quad\text{(activation)}\\[4pt]
&x_{ij}\ge 0 \qquad &&\forall i\in I,\; j\in J\\
&y_i\in\{0,1\} \qquad &&\forall i\in I
\end{aligned}
\]

---

### Data Mapping Rules

| Parameter | Source Table | Column(s) | Mapping |
|-----------|--------------|-----------|---------|
| \(f_i\) | `file_1_view_0` | `fixed_costs` | `f_i = fixed_costs` where `Unnamed: 0 = i` |
| \(t_{ij}\) | `file_2_view_0` | `C1`, `C2` | `t_{ij} = value` where `Unnamed: 0 = i` and column header = `j` |
| \(d_j\) | `file_0_view_0` | `demand` | `d_j = demand` where `customer = j` |

All index sets \(I\) and \(J\) are taken from the complete set of rows returned in the corresponding tables. No additional filtering or aggregation is applied. The model uses the exact numeric values from the source tables as parameters.