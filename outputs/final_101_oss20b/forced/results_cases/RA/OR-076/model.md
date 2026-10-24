**Mathematical model**

Let  

* \(W\) – set of warehouses (Warehouse ID)  
* \(C\) – set of customers (Customer ID)

| Symbol | Meaning | Domain |
|--------|---------|--------|
| \(y_w\) | binary variable: 1 if warehouse \(w\) is opened | \(\{0,1\}\) |
| \(x_{wc}\) | integer number of units shipped from warehouse \(w\) to customer \(c\) | \(\mathbb{Z}_{\ge 0}\) |

| Symbol | Parameter | Source |
|--------|-----------|--------|
| \(\text{FixedCost}_w\) | fixed annual opening cost of warehouse \(w\) | `file_1_view_0`, column `Fixed_Cost` |
| \(\text{Capacity}_w\) | maximum service capacity of warehouse \(w\) | `file_1_view_0`, column `Capacity` |
| \(\text{Demand}_c\) | demand of customer \(c\) | `file_2_view_0`, column `Demand` |
| \(\text{Cost}_{wc}\) | transportation cost per unit from \(w\) to \(c\) | `file_0_view_0`, row `Warehouse ID = w`, column `C{c}` (exact column‑id alignment) |

**Objective**

\[
\min \;\sum_{w\in W}\text{FixedCost}_w\,y_w
      +\sum_{w\in W}\sum_{c\in C}\text{Cost}_{wc}\,x_{wc}
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{w\in W}x_{wc}\;\ge\;\text{Demand}_c,\qquad \forall\,c\in C
   \]

2. **Warehouse capacity**  
   \[
   \sum_{c\in C}x_{wc}\;\le\;\text{Capacity}_w\,y_w,\qquad \forall\,w\in W
   \]

3. **Non‑negativity**  
   \[
   x_{wc}\;\ge\;0,\qquad \forall\,w\in W,\;c\in C
   \]

4. **Binary opening decision**  
   \[
   y_w\in\{0,1\},\qquad \forall\,w\in W
   \]

---

### Data Mapping Rules

| Parameter | Table ID | Key column | Value column | Mapping rule |
|-----------|----------|------------|--------------|--------------|
| \(\text{FixedCost}_w\) | `file_1_view_0` | `Warehouse ID` | `Fixed_Cost` | `FixedCost[w] = file_1_view_0[Warehouse ID = w][Fixed_Cost]` |
| \(\text{Capacity}_w\) | `file_1_view_0` | `Warehouse ID` | `Capacity` | `Capacity[w] = file_1_view_0[Warehouse ID = w][Capacity]` |
| \(\text{Demand}_c\) | `file_2_view_0` | `Customer ID` | `Demand` | `Demand[c] = file_2_view_0[Customer ID = c][Demand]` |
| \(\text{Cost}_{wc}\) | `file_0_view_0` | `Warehouse ID` | `C{c}` (exact column‑id alignment) | `Cost[w,c] = file_0_view_0[Warehouse ID = w][C{c}]` |

*Set definitions*  
- \(W =\) all `Warehouse ID` values in `file_1_view_0`.  
- \(C =\) all `Customer ID` values in `file_2_view_0`.

These mapping rules bind every parameter in the model to the corresponding source data without enumerating individual values.