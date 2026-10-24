**Mathematical Model**

Let  

- \(I\) be the set of products whose name starts with “Books”.  
- For each \(i\in I\)  
  - \(r_i\) : revenue per unit (from column **Revenue**)  
  - \(d_i\) : demand quantity (from column **Demand**)  
  - \(v_i\) : initial inventory (from column **Initial Inventory**)  
- Decision variable  
  - \(x_i\) : number of units of product \(i\) to be sold (integer, \(\ge 0\))

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le d_i && \forall i\in I \\[2pt]
             & 0 \le x_i \le v_i && \forall i\in I \\[2pt]
             & x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

```json
{
  "table_id": "file_0_view_0",
  "index_set": {
    "I": "Product_Name"
  },
  "parameters": {
    "r_i": "Revenue",
    "d_i": "Demand",
    "v_i": "Initial Inventory"
  },
  "variables": {
    "x_i": "Decision variable for product i"
  }
}
```
