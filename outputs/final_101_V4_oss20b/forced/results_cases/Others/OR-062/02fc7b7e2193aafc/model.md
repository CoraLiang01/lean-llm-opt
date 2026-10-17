## Mathematical Model

Let  

* \(I=\{i\}\) be the set of suppliers (rows of **fixed_cost.csv**).  
* \(J=\{j\}\) be the set of stores (columns of **transportation_costs.csv** except the first column).  

Parameters (all taken directly from the source tables)

| Symbol | Source table | Column | Description |
|--------|--------------|--------|-------------|
| \(f_i\) | file_1_view_0 | fixed_costs | Fixed opening cost of supplier \(i\) |
| \(c_{ij}\) | file_2_view_0 | column \(j\) | Transportation cost per unit from supplier \(i\) to store \(j\) |
| \(d_j\) | file_0_view_0 | demand | Demand (units) of store \(j\) |
| \(M\) | – | – | A sufficiently large constant, e.g. \(M=\sum_{j\in J} d_j\) |

Decision variables  

* \(y_i \in \{0,1\}\) – 1 if supplier \(i\) is opened, 0 otherwise.  
* \(x_{ij}\ge 0\) – quantity supplied from supplier \(i\) to store \(j\).

\[
\begin{aligned}
\min\;& \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{i\in I} x_{ij} \;=\; d_j \qquad &&\forall j\in J \quad\text{(demand satisfaction)}\\[4pt]
& x_{ij} \;\le\; M\,y_i \qquad &&\forall i\in I,\; j\in J \quad\text{(supplier activation)}\\[4pt]
& x_{ij}\;\ge\;0 \qquad &&\forall i\in I,\; j\in J\\[4pt]
& y_i\in\{0,1\} \qquad &&\forall i\in I
\end{aligned}
\]

---

## Data Mapping Rules

| Parameter | Table ID | Column | Notes |
|-----------|----------|--------|-------|
| \(f_i\) | file_1_view_0 | fixed_costs | Row identifier is the supplier name in column **Unnamed: 0** |
| \(c_{ij}\) | file_2_view_0 | column \(j\) | Row identifier is the supplier name in column **Unnamed: 0**; column names are the store names |
| \(d_j\) | file_0_view_0 | demand | Row identifier is the store name in column **Customer** |
| \(M\) | – | – | Computed as \(\sum_{j} d_j\) from the demand table |

The index sets \(I\) and \(J\) are taken directly from the row and column identifiers of the respective tables, with no additional filtering. All variables and constraints are defined over these sets as shown in the model.