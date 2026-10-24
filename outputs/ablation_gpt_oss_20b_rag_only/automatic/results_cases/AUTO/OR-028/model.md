**Abstract Optimization Model**

```ampl
/* Index set of products */
set P;

/* Parameters */
param rev{P};          /* revenue per unit – from column “Revenue” */
param dem{P};          /* demand – from column “Demand” */
param inv{P};          /* initial inventory – from column “Initial Inventory” */

/* Decision variables */
var x{P} integer >= 0; /* units of product p to fulfill */

/* Objective – maximize total revenue */
maximize TotalRevenue:
    sum {p in P} rev[p] * x[p];

/* Constraints */

/* Do not exceed the known demand for each product */
subject to DemandLimit {p in P}:
    x[p] <= dem[p];

/* Do not exceed the available initial inventory for each product */
subject to InventoryLimit {p in P}:
    x[p] <= inv[p];
```

---

**Data Mapping**

| Source | Table ID          | Column Name          | Mapped to |
|--------|-------------------|----------------------|-----------|
| WomenClothingEcommerceSalesData.csv | `file_0_view_0` | `Product Name` | `P` (index) |
| WomenClothingEcommerceSalesData.csv | `file_0_view_0` | `Revenue` | `rev` |
| WomenClothingEcommerceSalesData.csv | `file_0_view_0` | `Demand` | `dem` |
| WomenClothingEcommerceSalesData.csv | `file_0_view_0` | `Initial Inventory` | `inv` |

*All parameters and variables are defined solely from the data in the supplied table; no additional data or assumptions are introduced.*