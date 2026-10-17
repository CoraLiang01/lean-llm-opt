**Mathematical Model**

Let  

- \(I\) – set of all products (identified by *Product Name*).  
- \(x_i\) – production quantity of product \(i\) (continuous, \(x_i\ge 0\)).  

Parameters (all taken from the CSV file *file_0_view_0*)  

| Symbol | Meaning | Source column |
|--------|---------|---------------|
| \(L_i\) | Labor required per unit of product \(i\) | *Labor per unit* |
| \(M_i\) | Material required per unit of product \(i\) | *Material per unit* |
| \(P_i\) | Selling price per unit of product \(i\) | *Selling Price* |
| \(C_i\) | Variable cost per unit of product \(i\) | *Variable Cost* |

Constants  

- Weekly labor capacity: \(L^{\max}=1650\).  
- Weekly material capacity: \(M^{\max}=1850\).  
- Fixed operating cost per week: \(F=4500\).

---

### Objective  

\[
\max_{x}\; \sum_{i\in I}\bigl(P_i-C_i\bigr)\,x_i \;-\; F
\]

### Constraints  

1. **Labor capacity**  
   \[
   \sum_{i\in I} L_i\,x_i \;\le\; L^{\max}
   \]

2. **Material capacity**  
   \[
   \sum_{i\in I} M_i\,x_i \;\le\; M^{\max}
   \]

3. **Non‑negativity**  
   \[
   x_i \;\ge\; 0 \quad \forall i\in I
   \]

---

**Data Mapping**

| Parameter | Table ID | Column name |
|-----------|----------|-------------|
| \(L_i\) | `file_0_view_0` | `Labor per unit` |
| \(M_i\) | `file_0_view_0` | `Material per unit` |
| \(P_i\) | `file_0_view_0` | `Selling Price` |
| \(C_i\) | `file_0_view_0` | `Variable Cost` |

The index set \(I\) is implicitly defined by the distinct *Product Name* values present in the same table. No other data sources are used.