**Mathematical Model**

Let  

* \(I=\{S1,S2,S3,S4,S5,S6\}\) – suppliers  
* \(J=\{C1,C2,C3,C4,C5,C6\}\) – stores  

Parameters  

* \(f_i\) – fixed cost of opening supplier \(i\)  
  \(\displaystyle f_i =\) value in column **fixed_costs** of *fixed_cost.csv* for supplier \(i\).  
* \(t_{ij}\) – transportation cost per unit from supplier \(i\) to store \(j\)  
  \(\displaystyle t_{ij} =\) value in column \(j\) of row \(i\) in *transportation_costs.csv*.  
* \(d_j\) – demand of store \(j\)  
  \(\displaystyle d_j =\) value in column **demand** of *demand.csv* for store \(j\).  
* \(M=\sum_{j\in J} d_j =1080\) – a sufficiently large constant.

Decision variables  

* \(y_i\in\{0,1\}\) – 1 if supplier \(i\) is opened, 0 otherwise.  
* \(x_{ij}\ge 0\) – quantity shipped from supplier \(i\) to store \(j\).

Objective  

\[
\min \;\; \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} t_{ij}\,x_{ij}
\]

Constraints  

1. **Demand satisfaction** (exactly meet demand)  
   \[
   \sum_{i\in I} x_{ij}=d_j \qquad \forall j\in J
   \]

2. **Supplier activation** (no shipments if supplier closed)  
   \[
   x_{ij}\le M\,y_i \qquad \forall i\in I,\; j\in J
   \]

3. **Non‑negativity**  
   \[
   x_{ij}\ge 0 \qquad \forall i\in I,\; j\in J
   \]

4. **Binary**  
   \[
   y_i\in\{0,1\}\qquad \forall i\in I
   \]

---

### Data Mapping

| Parameter | Source Table | Column(s) | Mapping |
|-----------|--------------|-----------|---------|
| \(f_i\) | *fixed_cost.csv* (table_id **file_1_view_0**) | **fixed_costs** | \(f_{S1}=98.88,\; f_{S2}=99.73,\; f_{S3}=94.01,\; f_{S4}=93.77,\; f_{S5}=107.59,\; f_{S6}=112.65\) |
| \(t_{ij}\) | *transportation_costs.csv* (table_id **file_2_view_0**) | **C1**–**C6** | \(t_{S1,C1}=0.08,\; t_{S1,C2}=52.33,\dots\) (full 6×6 matrix) |
| \(d_j\) | *demand.csv* (table_id **file_0_view_0**) | **demand** | \(d_{C1}=216,\; d_{C2}=216,\; d_{C3}=216,\; d_{C4}=144,\; d_{C5}=144,\; d_{C6}=144\) |
| \(M\) | Derived | – | \(M=\sum_j d_j=1080\) |

All indices \(i\) and \(j\) are taken from the complete sets of suppliers and stores returned by the observation; no additional filtering is applied. The model uses the exact numeric values from the source files and respects the variable domains and constraints specified in the query.