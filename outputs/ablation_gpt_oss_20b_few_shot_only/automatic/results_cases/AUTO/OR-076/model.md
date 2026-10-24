**Mathematical Model**

Let  

* \(W=\{W1,\dots ,W10\}\) – set of potential warehouses  
* \(C=\{C1,\dots ,C20\}\) – set of customers  

Parameters  

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(f_w\) | Fixed annual opening cost of warehouse \(w\) | `warehouse.csv` – column **Fixed_Cost** |
| \(cap_w\) | Service capacity of warehouse \(w\) | `warehouse.csv` – column **Capacity** |
| \(d_c\) | Demand of customer \(c\) | `demand.csv` – column **Demand** |
| \(t_{w,c}\) | Unit transportation cost from warehouse \(w\) to customer \(c\) | `cost.csv` – cell at row \(w\), column \(c\) |
| \(M_w\) | Big‑\(M\) for warehouse \(w\) (e.g. \(M_w=\sum_{c\in C} d_c\)) | Derived |

Decision variables  

* \(y_w \in \{0,1\}\) – 1 if warehouse \(w\) is opened, 0 otherwise  
* \(x_{w,c}\ge 0\) – quantity shipped from \(w\) to \(c\)

Objective  

\[
\min \;\sum_{w\in W} f_w\,y_w \;+\; \sum_{w\in W}\sum_{c\in C} t_{w,c}\,x_{w,c}
\]

Constraints  

1. **Customer demand satisfaction**  
   \[
   \sum_{w\in W} x_{w,c}=d_c ,\qquad \forall c\in C
   \]

2. **Warehouse capacity**  
   \[
   \sum_{c\in C} x_{w,c}\le cap_w ,\qquad \forall w\in W
   \]

3. **Activation (big‑\(M\))**  
   \[
   x_{w,c}\le M_w\,y_w ,\qquad \forall w\in W,\; \forall c\in C
   \]

4. **Variable domains**  
   \[
   y_w\in\{0,1\},\; x_{w,c}\ge 0
   \]

---

**Data Mapping Rules**

| Parameter | File | Column(s) | Mapping |
|-----------|------|-----------|---------|
| \(f_w\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/warehouse.csv` | `Fixed_Cost` | `f_w = warehouse[Fixed_Cost]` |
| \(cap_w\) | same file | `Capacity` | `cap_w = warehouse[Capacity]` |
| \(d_c\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/demand.csv` | `Demand` | `d_c = demand[Demand]` |
| \(t_{w,c}\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/cost.csv` | row \(w\), column \(c\) | `t_{w,c} = cost[w][c]` |
| \(M_w\) | derived | – | `M_w = sum_{c∈C} d_c` (or any upper bound ≥ total demand) |

All indices, parameters, and variables are defined exactly as above; no numerical values are enumerated in the model.