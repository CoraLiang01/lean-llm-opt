**Mathematical Model**

Let  

* \(I=\{S_1,\dots ,S_{24}\}\) – suppliers  
* \(J=\{C_1,\dots ,C_{25}\}\) – customers  

Parameters  

* \(f_i\) – fixed opening cost of supplier \(i\) (from *fixed_cost.csv*)  
* \(c_{ij}\) – transportation cost per unit from supplier \(i\) to customer \(j\) (from *transportation_costs.csv*)  
* \(d_j\) – demand of customer \(j\) (from *demand.csv*)

Decision variables  

* \(y_i\in\{0,1\}\) – 1 if supplier \(i\) is opened, 0 otherwise  
* \(x_{ij}\ge 0\) – quantity shipped from supplier \(i\) to customer \(j\)

\[
\begin{aligned}
\min\;&\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\;&\sum_{i\in I} x_{ij}=d_j \qquad &&\forall j\in J \quad\text{(customer demand)}\\[4pt]
&x_{ij}\;\le\; d_j\,y_i \qquad &&\forall i\in I,\; j\in J \quad\text{(open‑supplier link)}\\[4pt]
&y_i\in\{0,1\} \qquad &&\forall i\in I\\
&x_{ij}\ge 0 \qquad &&\forall i\in I,\; j\in J
\end{aligned}
\]

---

### Data Mapping

| Parameter | Source Table | Column(s) | Mapping Rule |
|-----------|--------------|-----------|--------------|
| \(f_i\) | `file_1_view_0` (fixed_cost.csv) | `Unnamed: 0` → supplier id, `fixed_costs` → cost | For each row, set \(f_{\text{row.Unnamed: 0}} = \text{row.fixed_costs}\) |
| \(c_{ij}\) | `file_2_view_0` (transportation_costs.csv) | `Unnamed: 0` → supplier id, columns `C1`–`C25` → customer ids | For each row \(i\) and column \(j\), set \(c_{i,j} = \text{row.Cj}\) |
| \(d_j\) | `file_0_view_0` (demand.csv) | `customer` → customer id, `demand` → demand | For each row, set \(d_{\text{row.customer}} = \text{row.demand}\) |

*All indices \(i\) and \(j\) are taken from the full set of identifiers present in the respective tables; no additional filtering is applied.*