**Abstract Optimization Model**

```ampl
/* 1. Index set – all products in the data table  */
set I;                     /* product identifiers (Product Name) */

/* 2. Parameters – extracted from the source table  */
param revenue {I};         /* revenue per unit of product i */
param demand  {I};         /* maximum demand for product i */
param inventory {I};       /* initial inventory available for product i */

/* 3. Decision variables – units of each product to sell  */
var x {I} integer >= 0;    /* units of product i to fulfill */

/* 4. Objective – maximize total revenue  */
maximize TotalRevenue:
    sum {i in I} revenue[i] * x[i];

/* 5. Constraints  */

/* 5.1  Do not sell more than the known demand  */
subject to DemandLimit {i in I}:
    x[i] <= demand[i];

/* 5.2  Do not sell more than the available inventory  */
subject to InventoryLimit {i in I}:
    x[i] <= inventory[i];
```

---

**Data Mapping**

| Symbol | Source | Table ID | Column Name |
|--------|--------|----------|-------------|
| `I` | Index set | `file_0_view_0` | `Product Name` |
| `revenue[i]` | Parameter | `file_0_view_0` | `Revenue` |
| `demand[i]` | Parameter | `file_0_view_0` | `Demand` |
| `inventory[i]` | Parameter | `file_0_view_0` | `Initial Inventory` |
| `x[i]` | Decision variable | – | – |

*All rows of the table `file_0_view_0` are used to populate the index set `I` and the corresponding parameter values.*