**Mathematical Model**

Let  

* \(I=\{S_1,S_2,\dots ,S_{24}\}\) ( suppliers )  
* \(J=\{C_1,C_2,\dots ,C_{25}\}\) ( customers )  

Parameters  

* \(f_i\) fixed cost of opening supplier \(i\) \((i\in I)\)  
* \(c_{ij}\) transportation cost per unit from supplier \(i\) to customer \(j\) \((i\in I,\;j\in J)\)  
* \(d_j\) demand of customer \(j\) \((j\in J)\)  
* \(M=\sum_{j\in J} d_j\) (upper bound on total shipment from any supplier)

Decision variables  

* \(y_i\in\{0,1\}\) 1 if supplier \(i\) is opened, 0 otherwise  
* \(x_{ij}\ge 0\) quantity shipped from supplier \(i\) to customer \(j\)

---

**Objective**

\[
\min \;\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}\;+\;\sum_{i\in I} f_i\,y_i
\]

---

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}=d_j ,\qquad \forall j\in J
   \]

2. **Supplier activation (big‑\(M\) formulation)**  
   \[
   \sum_{j\in J} x_{ij}\;\le\; M\,y_i ,\qquad \forall i\in I
   \]

3. **Variable domains**  
   \[
   x_{ij}\ge 0,\; y_i\in\{0,1\}\quad \forall i\in I,\; j\in J
   \]

---

**Data Mapping**

| Parameter | Source file | Mapping rule |
|-----------|-------------|--------------|
| \(d_j\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/demand.csv` | `d_j = demand.csv[customer == j, demand]` |
| \(f_i\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/fixed_cost.csv` | `f_i = fixed_cost.csv[Unnamed: 0 == i, fixed_costs]` |
| \(c_{ij}\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/transportation_costs.csv` | `c_{ij} = transportation_costs.csv[Unnamed: 0 == i, Cj]` |
| \(M\) | computed | `M = sum_{j∈J} d_j` |

All identifiers (`S1`–`S24`, `C1`–`C25`) are taken verbatim from the corresponding CSV files. No numerical values are enumerated in the model; the parameters are linked to the data files as shown.