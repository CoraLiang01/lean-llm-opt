**Mathematical Model**

Let  

- \(S=\{s\mid s\text{ is a store id in } \text{file\_1\_view\_0}\}\)  
- \(C=\{c\mid c\text{ is a customer id in } \text{file\_0\_view\_0}\}\)

Parameters  

- \(d_c\) – demand of customer \(c\) (from column **demand** in **file_0_view_0**)  
- \(s_s\) – supply capacity of store \(s\) (from column **supply_capacity** in **file_1_view_0**)  
- \(c_{s,c}\) – transportation cost per unit from store \(s\) to customer \(c\)  
  (from the matrix in **file_2_view_0**, row id **Unnamed: 0** = \(s\), column id = \(c\))

Decision variables  

- \(x_{s,c}\ge 0\) – quantity shipped from store \(s\) to customer \(c\) (continuous)

Objective  

\[
\min \;\sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c}
\]

Constraints  

1. **Demand satisfaction** – for every customer \(c\in C\)

\[
\sum_{s\in S} x_{s,c}=d_c
\]

2. **Supply limits** – for every store \(s\in S\)

\[
\sum_{c\in C} x_{s,c}\le s_s
\]

3. **Non‑negativity**

\[
x_{s,c}\ge 0 \quad \forall s\in S,\;c\in C
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column(s) | Mapping Rule |
|-----------|--------------|------------------|--------------|
| \(S\) | `file_1_view_0` | `Unnamed: 0` | Set of all store identifiers |
| \(C\) | `file_0_view_0` | `customer` | Set of all customer identifiers |
| \(d_c\) | `file_0_view_0` | `demand` | For each customer \(c\), value in column `demand` |
| \(s_s\) | `file_1_view_0` | `supply_capacity` | For each store \(s\), value in column `supply_capacity` |
| \(c_{s,c}\) | `file_2_view_0` | matrix entries | For each store \(s\) (row id `Unnamed: 0`) and customer \(c\) (column id `customer`), value is the transportation cost |

All indices, parameters, and variables are defined solely from the data tables above; no additional rows or columns are introduced. The model is a linear program with continuous decision variables.