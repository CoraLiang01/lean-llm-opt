**Mathematical Model**

Let  

* \(I=\{S_1,\dots ,S_{24}\}\) – suppliers (from *fixed_cost.csv*).  
* \(J=\{C_1,\dots ,C_{25}\}\) – supermarkets (from *demand.csv*).  

Parameters  

* \(f_i\) – fixed opening cost of supplier \(i\) (from *fixed_cost.csv*).  
* \(d_j\) – demand of supermarket \(j\) (from *demand.csv*).  
* \(t_{ij}\) – transportation cost per unit from supplier \(i\) to supermarket \(j\) (from *transportation_costs.csv*).  

Decision variables  

* \(y_i\in\{0,1\}\) – 1 if supplier \(i\) is opened.  
* \(x_{ij}\ge 0\) – quantity supplied from \(i\) to \(j\).  

\[
\begin{aligned}
\min\;&\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} t_{ij}\,x_{ij} \\[4pt]
\text{s.t. } &\sum_{i\in I} x_{ij}=d_j , &&\forall j\in J \quad\text{(demand)}\\[2pt]
&x_{ij}\;\le\; d_j\,y_i , &&\forall i\in I,\; j\in J \quad\text{(supplier open)}\\[2pt]
&x_{ij}\;\ge\;0 , &&\forall i\in I,\; j\in J\\[2pt]
&y_i\in\{0,1\} , &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Source Column | Mapping |
|-----------|--------------|---------------|---------|
| \(f_i\) | `file_1_view_0` | `fixed_costs` | \(f_{S_k}= \text{fixed_costs of row }k\) |
| \(d_j\) | `file_0_view_0` | `demand` | \(d_{C_k}= \text{demand of row }k\) |
| \(t_{ij}\) | `file_2_view_0` | column `C_k` of row `S_i` | \(t_{S_i,C_k}= \text{value in cell (row }S_i,\text{ column }C_k)\) |

| Variable | Domain | Definition |
|----------|--------|------------|
| \(y_i\) | \(\{0,1\}\) | Binary open‑flag for supplier \(i\) |
| \(x_{ij}\) | \(\mathbb{R}_{+}\) | Quantity shipped from supplier \(i\) to supermarket \(j\) |

*Index sets*  
- \(I\) is the set of all supplier identifiers appearing in `file_1_view_0` (`Unnamed: 0`).  
- \(J\) is the set of all supermarket identifiers appearing in `file_0_view_0` (`customer`).  

All constraints and the objective use only these indices and the mapped parameters. No additional bounds or capacities are imposed beyond those derived from the data.