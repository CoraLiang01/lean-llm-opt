**Mathematical Model**

| Symbol | Meaning |
|--------|---------|
| \(I\) | Set of all “Organ” products (one element per returned row) |
| \(rev_i\) | Revenue per unit of product \(i\) |
| \(dem_i\) | Demand quantity of product \(i\) |
| \(inv_i\) | Initial inventory of product \(i\) |
| \(x_i\) | Units of product \(i\) to be sold (decision variable) |

**Objective**

\[
\max_{x}\;\;\sum_{i\in I} rev_i \; x_i
\]

**Constraints**

\[
\begin{aligned}
&0 \le x_i \le dem_i &&\forall i\in I \\
&0 \le x_i \le inv_i &&\forall i\in I
\end{aligned}
\]

(Equivalently \(x_i \le \min\{dem_i,\,inv_i\}\).)

**Data Mapping**

```json
{
  "table_id": "file_0_view_0",
  "columns": {
    "product_id": "Sub Category",
    "rev_i": "Revenue",
    "dem_i": "Demand",
    "inv_i": "Initial Inventory"
  },
  "index_set": "I"
}
```

*The index set \(I\) consists of the three rows returned by the filter “Sub Category contains 'Organ'”. No numerical values are enumerated; all parameters are linked directly to the corresponding CSV columns.*