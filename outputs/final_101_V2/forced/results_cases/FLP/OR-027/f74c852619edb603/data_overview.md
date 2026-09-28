Here are all the products classified under ‘Organ’ (i.e., those with "Organic" in their sub-category), along with their ‘Revenue’, ‘Initial Inventory’, and ‘Demand’:

| Sub Category         | Revenue | Initial Inventory | Demand  |
|---------------------|---------|------------------|---------|
| Organic Fruits      | 60.8    | 5,034,020.0      | 678,906 |
| Organic Staples     | 918.45  | 5,589,290.0      | 749,927 |
| Organic Vegetables  | 77.52   | 5,202,710.0      | 699,808 |

**Variables:**
- \( x_1 \): Units of Organic Fruits fulfilled
- \( x_2 \): Units of Organic Staples fulfilled
- \( x_3 \): Units of Organic Vegetables fulfilled

**Data Table:**

| Product             | Revenue per unit | Initial Inventory | Demand  |
|---------------------|------------------|------------------|---------|
| Organic Fruits      | 60.8             | 5,034,020        | 678,906 |
| Organic Staples     | 918.45           | 5,589,290        | 749,927 |
| Organic Vegetables  | 77.52            | 5,202,710        | 699,808 |

**Decision variables:**
- \( x_1 \leq \min(5,034,020, 678,906) \)
- \( x_2 \leq \min(5,589,290, 749,927) \)
- \( x_3 \leq \min(5,202,710, 699,808) \)

**Objective:**
Maximize total revenue:
\[
\text{Maximize } 60.8x_1 + 918.45x_2 + 77.52x_3
\]
subject to inventory and demand constraints for each organic product.