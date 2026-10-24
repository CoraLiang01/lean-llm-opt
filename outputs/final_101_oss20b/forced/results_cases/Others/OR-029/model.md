**Mathematical Model**

Let  

* \(I\) be the set of all FAUX products that appear in the filtered view  
  `file_0_view_0`.  
* For each \(i\in I\) define the decision variable  

\[
x_i \in \mathbb{Z}_{+}\quad\text{(units of product }i\text{ to sell)} .
\]

**Parameters** (all taken directly from the view)

| Symbol | Source | Column |
|--------|--------|--------|
| \(r_i\) | `file_0_view_0` | `Revenue` |
| \(d_i\) | `file_0_view_0` | `Demand` |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` |

**Objective**

\[
\max \;\; Z = \sum_{i\in I} r_i\,x_i
\]

**Constraints**

1. **Demand limit**  
   \[
   x_i \;\le\; d_i \qquad \forall i\in I
   \]

2. **Inventory limit**  
   \[
   x_i \;\le\; s_i \qquad \forall i\in I
   \]

3. **Non‑negativity**  
   \[
   x_i \;\ge\; 0 \qquad \forall i\in I
   \]

---

### Data Mapping

| Table ID | Column | Parameter |
|----------|--------|-----------|
| `file_0_view_0` | `Revenue` | \(r_i\) |
| `file_0_view_0` | `Demand` | \(d_i\) |
| `file_0_view_0` | `Initial Inventory` | \(s_i\) |

The index set \(I\) consists of all rows returned by the filter  
(`Product Name` contains “FAUX”) in `file_0_view_0`. No additional rows are added.