**Mathematical Model**

\[
\begin{aligned}
\text{minimize}\quad & Z = \sum_{s\in S}\sum_{c\in C} \text{Cost}_{s,c}\;x_{s,c} \\[4pt]
\text{subject to}\quad
& \sum_{c\in C} x_{s,c} \;\le\; \text{SupplyCap}_{s} && \forall s\in S \\[4pt]
& \sum_{s\in S} x_{s,c} \;=\; \text{Demand}_{c} && \forall c\in C \\[4pt]
& x_{s,c}\;\ge\;0 && \forall s\in S,\;c\in C
\end{aligned}
\]

**Index Sets**

- \(S\): set of suppliers (identified by the values in column **Unnamed: 0** of *file_1_view_0*).  
- \(C\): set of customers (identified by the values in column **customer** of *file_0_view_0*).

**Parameters**

| Symbol | Description | Source table & column |
|--------|-------------|-----------------------|
| \(\text{SupplyCap}_{s}\) | Daily supply capacity of supplier \(s\) | *file_1_view_0*, column **supply_capacity** |
| \(\text{Demand}_{c}\) | Daily demand of customer \(c\) | *file_0_view_0*, column **demand** |
| \(\text{Cost}_{s,c}\) | Transportation cost per unit from supplier \(s\) to customer \(c\) | *file_2_view_0*, columns **C1**, **C2**, …, **C10** (one column per customer) |

**Variables**

- \(x_{s,c}\) – quantity shipped from supplier \(s\) to customer \(c\) (continuous, non‑negative).

**Data Mapping Rules**

1. **Suppliers**  
   - Table ID: `file_1_view_0`  
   - Key column: `Unnamed: 0`  
   - Parameter column: `supply_capacity`

2. **Customers**  
   - Table ID: `file_0_view_0`  
   - Key column: `customer`  
   - Parameter column: `demand`

3. **Transportation Costs**  
   - Table ID: `file_2_view_0`  
   - Key column: `Unnamed: 0` (matches suppliers)  
   - For each customer \(c\), the cost column is named exactly as the customer identifier (e.g., `C1`, `C2`, …).  

All parameters are mapped directly from the corresponding table columns; no additional data manipulation or enumeration of numeric values is required.