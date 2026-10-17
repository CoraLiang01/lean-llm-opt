**Mathematical Model**

Let  

* \(I\) – set of suppliers (indexed by the values in column **Unnamed: 0** of *fixed_cost.csv*).  
* \(J\) – set of stores (indexed by the row numbers 1–5 of *demand.csv*).  

Parameters  

* \(f_i\) – fixed opening cost of supplier \(i\)  
  \(\displaystyle f_i = \text{fixed_costs}_{i}\)  (table **file_1_view_0**, column **fixed_costs**).  

* \(d_j\) – demand of store \(j\)  
  \(\displaystyle d_j = \text{demand}_{j}\)  (table **file_0_view_0**, column **demand**).  

* \(c_{ij}\) – transportation cost per unit from supplier \(i\) to store \(j\)  
  \(\displaystyle c_{ij} = \text{transportation_costs}_{i,j}\)  
  (table **file_2_view_0**; column mapping:  
  \(j=1\!\rightarrow\! \text{CLARINDA}\),  
  \(j=2\!\rightarrow\! \text{FORT MADISON}\),  
  \(j=3\!\rightarrow\! \text{SIOUX CITY}\),  
  \(j=4\!\rightarrow\! \text{TOLEDO}\),  
  \(j=5\!\rightarrow\! \text{BANCROFT}\)).  

Decision variables  

* \(y_i \in \{0,1\}\) – 1 if supplier \(i\) is opened, 0 otherwise.  
* \(x_{ij} \ge 0\) – quantity shipped from supplier \(i\) to store \(j\).  

**Objective**

\[
\min \;\; \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij} \;\ge\; d_j , \qquad \forall j\in J
   \]

2. **Supplier activation linking**  
   \[
   x_{ij} \;\le\; d_j\,y_i , \qquad \forall i\in I,\; j\in J
   \]

3. **Non‑negativity**  
   \[
   x_{ij} \;\ge\; 0 , \qquad \forall i\in I,\; j\in J
   \]

4. **Binary supplier variables**  
   \[
   y_i \in \{0,1\} , \qquad \forall i\in I
   \]

---

### Data Mapping

| Parameter | Table ID | Column(s) | Notes |
|-----------|----------|-----------|-------|
| \(f_i\) | **file_1_view_0** | **fixed_costs** | Supplier identifier is the row’s **Unnamed: 0** value |
| \(d_j\) | **file_0_view_0** | **demand** | Store identifier is the row number (1–5) |
| \(c_{ij}\) | **file_2_view_0** | **CLARINDA**, **FORT MADISON**, **SIOUX CITY**, **TOLEDO**, **BANCROFT** | Column order maps to store indices 1–5 respectively; supplier identifier is the row’s **Unnamed: 0** value |

All other data are derived directly from these tables; no additional parameters are introduced.