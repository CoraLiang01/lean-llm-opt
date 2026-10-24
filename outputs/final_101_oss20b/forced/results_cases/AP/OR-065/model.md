**Symbolic Model**

Let  

* \(I=\{i\}\) be the set of warehouses (indexed by the values in column **Unnamed: 0** of *fixed_cost.csv*).  
* \(J=\{j\}\) be the set of musicians/bands (indexed by the values in column **customer** of *demand.csv*).  

Parameters  

* \(f_i\) – fixed cost of opening warehouse \(i\)  
  \(\displaystyle f_i = \text{fixed_costs}_{\text{file_1_view_0}}\)  
* \(t_{ij}\) – transportation cost per unit from warehouse \(i\) to customer \(j\)  
  \(\displaystyle t_{ij} = \text{C}_{j,\text{file_2_view_0}}\)  
* \(d_j\) – demand of customer \(j\)  
  \(\displaystyle d_j = \text{demand}_{\text{file_0_view_0}}\)

Decision variables  

* \(y_i \in \{0,1\}\) – 1 if warehouse \(i\) is activated, 0 otherwise.  
* \(x_{ij} \ge 0\) – quantity supplied from warehouse \(i\) to customer \(j\).

---

\[
\begin{aligned}
\min\;& \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} t_{ij}\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{i\in I} x_{ij} \;=\; d_j \qquad &&\forall j\in J \\[4pt]
& x_{ij} \;\le\; d_j\,y_i \qquad &&\forall i\in I,\; j\in J \\[4pt]
& y_i \in \{0,1\} \qquad &&\forall i\in I \\[2pt]
& x_{ij} \ge 0 \qquad &&\forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column | Index Column |
|-----------|--------------|---------------|--------------|
| \(f_i\)   | file_1_view_0 | fixed_costs   | Unnamed: 0   |
| \(t_{ij}\)| file_2_view_0 | C1, C2, C3    | Unnamed: 0   |
| \(d_j\)   | file_0_view_0 | demand        | customer     |

*No numerical values are listed; the model references the data via the table identifiers and column names above.*