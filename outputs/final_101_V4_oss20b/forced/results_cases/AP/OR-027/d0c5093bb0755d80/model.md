**Mathematical Model**

Let  

- \(I\) be the set of all products whose *Sub Category* contains the string “Organ”.  
- For each \(i\in I\)  

  \[
  \begin{aligned}
  &\text{rev}_i &&\text{revenue per unit of product }i \quad(\text{from column } \text{Revenue})\\
  &\text{dem}_i &&\text{demand quantity of product }i \quad(\text{from column } \text{Demand})\\
  &\text{inv}_i &&\text{initial inventory of product }i \quad(\text{from column } \text{Initial Inventory})\\
  &x_i &&\text{units of product }i \text{ to be sold}
  \end{aligned}
  \]

**Objective**

\[
\max_{x}\;\;\sum_{i\in I}\text{rev}_i\,x_i
\]

**Constraints**

\[
\begin{aligned}
&x_i \;\le\; \text{inv}_i &&\forall i\in I \quad(\text{inventory limit})\\
&x_i \;\le\; \text{dem}_i &&\forall i\in I \quad(\text{demand limit})\\
&x_i \;\ge\; 0 &&\forall i\in I\\
&x_i \in \mathbb{Z} &&\forall i\in I \quad(\text{units are integer})
\end{aligned}
\]

---

**Data Mapping**

```json
{
  "table_id": "file_0_view_0",
  "index": {
    "i": "Sub Category"
  },
  "parameters": {
    "rev_i": "Revenue",
    "dem_i": "Demand",
    "inv_i": "Initial Inventory"
  }
}
```

This mapping links each symbolic parameter to the corresponding column in the source CSV file `SupermartGrocerySales-RetailAnalyticsDataset.csv`. No numerical values are enumerated; the model remains symbolic and ready for numerical instantiation.