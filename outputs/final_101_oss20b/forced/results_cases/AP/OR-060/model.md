**Mathematical Model**

Let  

* \(S\) – set of suppliers, indexed by \(i\).  
* \(C\) – set of supermarkets, indexed by \(j\).

**Parameters**

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(f_i\) | Fixed opening cost of supplier \(i\) | `file_1_view_0` column **fixed_costs** (row id **Unnamed: 0** = \(i\)) |
| \(d_j\) | Demand of supermarket \(j\) | `file_0_view_0` column **demand** (row id **customer** = \(j\)) |
| \(t_{ij}\) | Transportation cost per unit from supplier \(i\) to supermarket \(j\) | `file_2_view_0` matrix entry (row id **Unnamed: 0** = \(i\), column id **customer** = \(j\)) |

**Decision Variables**

* \(y_i \in \{0,1\}\) – 1 if supplier \(i\) is opened, 0 otherwise.  
* \(x_{ij} \ge 0\) – quantity shipped from supplier \(i\) to supermarket \(j\).

**Objective**

\[
\min \;\; \sum_{i\in S} f_i\,y_i \;+\; \sum_{i\in S}\sum_{j\in C} t_{ij}\,x_{ij}
\]

**Constraints**

1. **Demand satisfaction**

\[
\sum_{i\in S} x_{ij} \;=\; d_j \quad \forall j\in C
\]

2. **Supplier activation**

\[
x_{ij} \;\le\; d_j\,y_i \quad \forall i\in S,\; j\in C
\]

3. **Variable domains**

\[
y_i \in \{0,1\}\;\;\forall i\in S,\qquad
x_{ij}\ge 0\;\;\forall i\in S,\; j\in C
\]

---

**Data Mapping Rules**

| Parameter | Table ID | Row key | Column key |
|-----------|----------|---------|------------|
| \(f_i\) | `file_1_view_0` | `Unnamed: 0` = supplier id | `fixed_costs` |
| \(d_j\) | `file_0_view_0` | `customer` = supermarket id | `demand` |
| \(t_{ij}\) | `file_2_view_0` | `Unnamed: 0` = supplier id | `customer` = supermarket id |

All indices \(i\) and \(j\) are taken from the full set of suppliers and supermarkets present in the respective tables. No additional constraints or bounds are imposed beyond those listed.