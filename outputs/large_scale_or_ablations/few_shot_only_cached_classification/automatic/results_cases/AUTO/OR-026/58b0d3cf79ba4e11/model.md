##### Sets and Indices

Let $\mathcal{F}$ be the set of all products classified as ‘Fashion’, indexed by $i$ in source order.

##### Parameters (from retrieved data, in source order)

For each $i \in \mathcal{F}$:

- $A_i$: Revenue for product $i$
- $d_i$: Demand for product $i$
- $I_i$: Initial Inventory for product $i$

| $i$ (Product Name)                  | $A_i$ (Revenue) | $d_i$ (Demand) | $I_i$ (Initial Inventory) |
|-------------------------------------|-----------------|---------------|--------------------------|
| Fashion accessories_10.18           | 10.18           | 12.0          | 80.0                     |
| Fashion accessories_12.09           | 12.09           | 2.0           | 10.0                     |
| Fashion accessories_12.19           | 12.19           | 10.0          | 80.0                     |
| Fashion accessories_12.54           | 12.54           | 2.0           | 10.0                     |
| Fashion accessories_12.78           | 12.78           | 2.0           | 10.0                     |
| Fashion accessories_14.48           | 14.48           | 5.0           | 40.0                     |
| Fashion accessories_15.43           | 15.43           | 2.0           | 10.0                     |
| Fashion accessories_15.5            | 15.5            | 2.0           | 10.0                     |
| Fashion accessories_15.62           | 15.62           | 11.0          | 80.0                     |
| Fashion accessories_16.28           | 16.28           | 2.0           | 10.0                     |
| Fashion accessories_16.45           | 16.45           | 6.0           | 40.0                     |
| Fashion accessories_17.48           | 17.48           | 9.0           | 60.0                     |
| Fashion accessories_17.49           | 17.49           | 14.0          | 100.0                    |
| Fashion accessories_17.87           | 17.87           | 6.0           | 40.0                     |
| Fashion accessories_17.94           | 17.94           | 8.0           | 50.0                     |
| Fashion accessories_18.08           | 18.08           | 6.0           | 40.0                     |
| Fashion accessories_19.66           | 19.66           | 14.0          | 100.0                    |
| Fashion accessories_19.7            | 19.7            | 2.0           | 10.0                     |
| Fashion accessories_19.77           | 19.77           | 14.0          | 100.0                    |
| Fashion accessories_20.01           | 20.01           | 11.0          | 90.0                     |
| Fashion accessories_21.32           | 21.32           | 2.0           | 10.0                     |
| Fashion accessories_21.48           | 21.48           | 3.0           | 20.0                     |
| Fashion accessories_21.94           | 21.94           | 8.0           | 50.0                     |
| Fashion accessories_22.32           | 22.32           | 10.0          | 80.0                     |
| Fashion accessories_22.51           | 22.51           | 10.0          | 70.0                     |
| Fashion accessories_23.82           | 23.82           | 7.0           | 50.0                     |
| Fashion accessories_25.42           | 25.42           | 11.0          | 80.0                     |
| Fashion accessories_25.56           | 25.56           | 11.0          | 70.0                     |
| Fashion accessories_27.02           | 27.02           | 5.0           | 30.0                     |
| Fashion accessories_27.18           | 27.18           | 3.0           | 20.0                     |
| Fashion accessories_27.38           | 27.38           | 8.0           | 60.0                     |
| Fashion accessories_29.42           | 29.42           | 13.0          | 100.0                    |
| Fashion accessories_29.56           | 29.56           | 7.0           | 50.0                     |
| Fashion accessories_30.14           | 30.14           | 14.0          | 100.0                    |
| Fashion accessories_30.37           | 30.37           | 4.0           | 30.0                     |
| Fashion accessories_30.61           | 30.61           | 2.0           | 10.0                     |
| Fashion accessories_30.62           | 30.62           | 2.0           | 10.0                     |
| Fashion accessories_31.73           | 31.73           | 14.0          | 90.0                     |
| Fashion accessories_31.9            | 31.9            | 2.0           | 10.0                     |
| Fashion accessories_32.62           | 32.62           | 6.0           | 40.0                     |
| Fashion accessories_33.52           | 33.52           | 2.0           | 10.0                     |
| Fashion accessories_33.63           | 33.63           | 2.0           | 10.0                     |
| Fashion accessories_34.7            | 34.7            | 3.0           | 20.0                     |
| Fashion accessories_35.19           | 35.19           | 14.0          | 100.0                    |
| Fashion accessories_36.51           | 36.51           | 12.0          | 90.0                     |
| Fashion accessories_36.85           | 36.85           | 7.0           | 50.0                     |
| Fashion accessories_37.15           | 37.15           | 6.0           | 40.0                     |
| Fashion accessories_37.55           | 37.55           | 13.0          | 100.0                    |
| Fashion accessories_37.95           | 37.95           | 14.0          | 100.0                    |

##### Decision Variables

For each $i \in \mathcal{F}$:

- $x_i$: Number of units of product $i$ to fulfill

##### Mathematical Model

**Objective:**

$$
\max \sum_{i \in \mathcal{F}} A_i \cdot x_i
$$

**Subject to:**

- Inventory constraints:
  $$
  x_i \leq I_i \qquad \forall i \in \mathcal{F}
  $$
- Demand constraints:
  $$
  x_i \leq d_i \qquad \forall i \in \mathcal{F}
  $$
- Non-negativity and integrality:
  $$
  x_i \in \mathbb{Z}_+, \qquad x_i \geq 0 \qquad \forall i \in \mathcal{F}
  $$

##### Retrieved Information

{
  "fashion_products": [
    {"Product Name": "Fashion accessories_10.18", "Revenue": 10.18, "Demand": 12.0, "Initial Inventory": 80.0},
    {"Product Name": "Fashion accessories_12.09", "Revenue": 12.09, "Demand": 2.0, "Initial Inventory": 10.0},
    {"Product Name": "Fashion accessories_12.19", "Revenue": 12.19, "Demand": 10.0, "Initial Inventory": 80.0},
    {"Product Name": "Fashion accessories_12.54", "Revenue": 12.54, "Demand": 2.0, "Initial Inventory": 10.0},
    {"Product Name": "Fashion accessories_12.78", "Revenue": 12.78, "Demand": 2.0, "Initial Inventory": 10.0},
    {"Product Name": "Fashion accessories_14.48", "Revenue": 14.48, "Demand": 5.0, "Initial Inventory": 40.0},
    {"Product Name": "Fashion accessories_15.43", "Revenue": 15.43, "Demand": 2.0, "Initial Inventory": 10.0},
    {"Product Name": "Fashion accessories_15.5", "Revenue": 15.5, "Demand": 2.0, "Initial Inventory": 10.0},
    {"Product Name": "Fashion accessories_15.62", "Revenue": 15.62, "Demand": 11.0, "Initial Inventory": 80.0},
    {"Product Name": "Fashion accessories_16.28", "Revenue": 16.28, "Demand": 2.0, "Initial Inventory": 10.0},
    {"Product Name": "Fashion accessories_16.45", "Revenue": 16.45, "Demand": 6.0, "Initial Inventory": 40.0},
    {"Product Name": "Fashion accessories_17.48", "Revenue": 17.48, "Demand": 9.0, "Initial Inventory": 60.0},
    {"Product Name": "Fashion accessories_17.49", "Revenue": 17.49, "Demand": 14.0, "Initial Inventory": 100.0},
    {"Product Name": "Fashion accessories_17.87", "Revenue": 17.87, "Demand": 6.0, "Initial Inventory": 40.0},
    {"Product Name": "Fashion accessories_17.94", "Revenue": 17.94, "Demand": 8.0, "Initial Inventory": 50.0},
    {"Product Name": "Fashion accessories_18.08", "Revenue": 18.08, "Demand": 6.0, "Initial Inventory": 40.0},
    {"Product Name": "Fashion accessories_19.66", "Revenue": 19.66, "Demand": 14.0, "Initial Inventory": 100.0},
    {"Product Name": "Fashion accessories_19.7", "Revenue": 19.7, "Demand": 2.0, "Initial Inventory": 10.0},
    {"Product Name": "Fashion accessories_19.77", "Revenue": 19.77, "Demand": 14.0, "Initial Inventory": 100.0},
    {"Product Name": "Fashion accessories_20.01", "Revenue": 20.01, "Demand": 11.0, "Initial Inventory": 90.0},
    {"Product Name": "Fashion accessories_21.32", "Revenue": 21.32, "Demand": 2.0, "Initial Inventory": 10.0},
    {"Product Name": "Fashion accessories_21.48", "Revenue": 21.48, "Demand": 3.0, "Initial Inventory": 20.0},
    {"Product Name": "Fashion accessories_21.94", "Revenue": 21.94, "Demand": 8.0, "Initial Inventory": 50.0},
    {"Product Name": "Fashion accessories_22.32", "Revenue": 22.32, "Demand": 10.0, "Initial Inventory": 80.0},
    {"Product Name": "Fashion accessories_22.51", "Revenue": 22.51, "Demand": 10.0, "Initial Inventory": 70.0},
    {"Product Name": "Fashion accessories_23.82", "Revenue": 23.82, "Demand": 7.0, "Initial Inventory": 50.0},
    {"Product Name": "Fashion accessories_25.42", "Revenue": 25.42, "Demand": 11.0, "Initial Inventory": 80.0},
    {"Product Name": "Fashion accessories_25.56", "Revenue": 25.56, "Demand": 11.0, "Initial Inventory": 70.0},
    {"Product Name": "Fashion accessories_27.02", "Revenue": 27.02, "Demand": 5.0, "Initial Inventory": 30.0},
    {"Product Name": "Fashion accessories_27.18", "Revenue": 27.18, "Demand": 3.0, "Initial Inventory": 20.0},
    {"Product Name": "Fashion accessories_27.38", "Revenue": 27.38, "Demand": 8.0, "Initial Inventory": 60.0},
    {"Product Name": "Fashion accessories_29.42", "Revenue": 29.42, "Demand": 13.0, "Initial Inventory": 100.0},
    {"Product Name": "Fashion accessories_29.56", "Revenue": 29.56, "Demand": 7.0, "Initial Inventory": 50.0},
    {"Product Name": "Fashion accessories_30.14", "Revenue": 30.14, "Demand": 14.0, "Initial Inventory": 100.0},
    {"Product Name": "Fashion accessories_30.37", "Revenue": 30.37, "Demand": 4.0, "Initial Inventory": 30.0},
    {"Product Name": "Fashion accessories_30.61", "Revenue": 30.61, "Demand": 2.0, "Initial Inventory": 10.0},
    {"Product Name": "Fashion accessories_30.62", "Revenue": 30.62, "Demand": 2.0, "Initial Inventory": 10.0},
    {"Product Name": "Fashion accessories_31.73", "Revenue": 31.73, "Demand": 14.0, "Initial Inventory": 90.0},
    {"Product Name": "Fashion accessories_31.9", "Revenue": 31.9, "Demand": 2.0, "Initial Inventory": 10.0},
    {"Product Name": "Fashion accessories_32.62", "Revenue": 32.62, "Demand": 6.0, "Initial Inventory": 40.0},
    {"Product Name": "Fashion accessories_33.52", "Revenue": 33.52, "Demand": 2.0, "Initial Inventory": 10.0},
    {"Product Name": "Fashion accessories_33.63", "Revenue": 33.63, "Demand": 2.0, "Initial Inventory": 10.0},
    {"Product Name": "Fashion accessories_34.7", "Revenue": 34.7, "Demand": 3.0, "Initial Inventory": 20.0},
    {"Product Name": "Fashion accessories_35.19", "Revenue": 35.19, "Demand": 14.0, "Initial Inventory": 100.0},
    {"Product Name": "Fashion accessories_36.51", "Revenue": 36.51, "Demand": 12.0, "Initial Inventory": 90.0},
    {"Product Name": "Fashion accessories_36.85", "Revenue": 36.85, "Demand": 7.0, "Initial Inventory": 50.0},
    {"Product Name": "Fashion accessories_37.15", "Revenue": 37.15, "Demand": 6.0, "Initial Inventory": 40.0},
    {"Product Name": "Fashion accessories_37.55", "Revenue": 37.55, "Demand": 13.0, "Initial Inventory": 100.0},
    {"Product Name": "Fashion accessories_37.95", "Revenue": 37.95, "Demand": 14.0, "Initial Inventory": 100.0}
  ]
}

**All coefficients, identifiers, and constraints are preserved in source order as required.**