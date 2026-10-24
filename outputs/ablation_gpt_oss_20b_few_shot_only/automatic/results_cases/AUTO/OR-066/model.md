**Symbolic Model**

Let  

* \(I=\{i\}\) – set of suppliers (IDs from *fixed_cost.csv* first column)  
* \(J=\{j\}\) – set of supermarkets (IDs from *demand.csv* first column)  

Parameters  

* \(f_i\) – fixed activation cost of supplier \(i\)  
  *source*: `Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv`, column **fixed_costs**  
* \(c_{ij}\) – unit transportation cost from supplier \(i\) to supermarket \(j\)  
  *source*: `Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv`, row‑index column **Unnamed: 0**, column \(j\)  
* \(d_j\) – demand of supermarket \(j\)  
  *source*: `Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv`, column **demand**  

Let  

* \(M = \sum_{j\in J} d_j\)  (big‑\(M\) upper bound on total shipment from any supplier)

Decision variables  

* \(x_{ij}\ge 0\) – quantity shipped from supplier \(i\) to supermarket \(j\)  
* \(y_i\in\{0,1\}\) – 1 if supplier \(i\) is activated, 0 otherwise  

**Objective**

\[
\min \;\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}\;+\;\sum_{i\in I} f_i\,y_i
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij} = d_j \quad \forall j\in J
   \]

2. **Activation linkage**  
   \[
   \sum_{j\in J} x_{ij} \;\le\; M\,y_i \quad \forall i\in I
   \]

3. **Variable domains**  
   \[
   x_{ij}\ge 0,\qquad y_i\in\{0,1\}\quad \forall i\in I,\; j\in J
   \]

---

**Data Mapping Rules**

| Parameter | Source File | Source Column | Index Column | Notes |
|-----------|-------------|---------------|--------------|-------|
| \(f_i\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv` | `fixed_costs` | `Unnamed: 0` | Supplier ID |
| \(c_{ij}\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv` | Column \(j\) (C1, C2, …) | `Unnamed: 0` | Supplier ID |
| \(d_j\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv` | `demand` | `customer` | Supermarket ID |

All indices \(i\) and \(j\) are taken directly from the respective first columns of the CSV files. No additional constraints or capacities are imposed beyond those listed.