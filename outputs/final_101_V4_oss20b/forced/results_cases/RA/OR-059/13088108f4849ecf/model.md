**Mathematical Model**

\[
\begin{aligned}
\min\;&\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\;&\sum_{i\in I} x_{ij} \;\ge\; d_j \qquad &&\forall j\in J \quad\text{(demand)}\\[4pt]
&x_{ij}\;\le\; M\,y_i \qquad &&\forall i\in I,\; j\in J \quad\text{(linking)}\\[4pt]
&y_i \in \{0,1\} \qquad &&\forall i\in I\\
&x_{ij}\;\ge\;0 \qquad &&\forall i\in I,\; j\in J
\end{aligned}
\]

where  

\[
M \;=\; \sum_{j\in J} d_j
\]

is a sufficiently large constant.

---

### Index Sets
| Symbol | Definition | Source |
|--------|------------|--------|
| \(I\) | Set of suppliers \(\{S1,\dots,S8\}\) | `file_1_view_0` column **Unnamed: 0** |
| \(J\) | Set of customers \(\{C1,\dots,C9\}\) | `file_0_view_0` column **customer** |

---

### Parameters
| Symbol | Meaning | Data Mapping |
|--------|---------|--------------|
| \(f_i\) | Fixed opening cost of supplier \(i\) | `file_1_view_0` column **fixed_costs** (indexed by **Unnamed: 0**) |
| \(d_j\) | Demand of customer \(j\) | `file_0_view_0` column **demand** (indexed by **customer**) |
| \(c_{ij}\) | Transportation cost per vehicle from supplier \(i\) to customer \(j\) | `file_2_view_0` matrix: row id **Unnamed: 0** = \(i\), column id = \(j\) (columns **C1**–**C9**) |

---

### Decision Variables
| Symbol | Domain | Interpretation |
|--------|--------|----------------|
| \(y_i\) | \(\{0,1\}\) | 1 if supplier \(i\) is opened, 0 otherwise |
| \(x_{ij}\) | \(\mathbb{R}_{\ge 0}\) | Number of vehicles shipped from supplier \(i\) to customer \(j\) |

---

**Data Mapping Rules**

1. **Suppliers**  
   - Table ID: `file_1_view_0`  
   - Index column: `Unnamed: 0`  
   - Parameter `f_i` maps to column `fixed_costs`.

2. **Customers**  
   - Table ID: `file_0_view_0`  
   - Index column: `customer`  
   - Parameter `d_j` maps to column `demand`.

3. **Transportation Costs**  
   - Table ID: `file_2_view_0`  
   - Row index column: `Unnamed: 0` (supplier)  
   - Column index: `C1`–`C9` (customers)  
   - Parameter `c_{ij}` maps to the corresponding cell value.

These mappings ensure that every parameter in the model is directly linked to the observed CSV data. No additional data or assumptions are introduced beyond what is supplied.