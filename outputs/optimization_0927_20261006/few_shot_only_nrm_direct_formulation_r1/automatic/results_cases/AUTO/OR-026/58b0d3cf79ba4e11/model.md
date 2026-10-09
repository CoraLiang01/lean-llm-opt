**Index Set:**  
Let $\mathcal{F}$ be the set of all products with 'Fashion' in the "Product Name" field, in source order:
- $\mathcal{F} = \{$  
  "Fashion accessories_10.18",  
  "Fashion accessories_12.09",  
  "Fashion accessories_12.19",  
  "Fashion accessories_12.54",  
  "Fashion accessories_12.78",  
  "Fashion accessories_14.48",  
  "Fashion accessories_15.43",  
  "Fashion accessories_15.5",  
  "Fashion accessories_15.62",  
  "Fashion accessories_16.28",  
  "Fashion accessories_16.45",  
  "Fashion accessories_17.48",  
  "Fashion accessories_17.49",  
  "Fashion accessories_17.87",  
  "Fashion accessories_17.94",  
  "Fashion accessories_18.08",  
  "Fashion accessories_19.66",  
  "Fashion accessories_19.7",  
  "Fashion accessories_19.77",  
  "Fashion accessories_20.01",  
  "Fashion accessories_21.32",  
  "Fashion accessories_21.48",  
  "Fashion accessories_21.94",  
  "Fashion accessories_22.32",  
  "Fashion accessories_22.51",  
  "Fashion accessories_23.82",  
  "Fashion accessories_25.42",  
  "Fashion accessories_25.56",  
  "Fashion accessories_27.02",  
  "Fashion accessories_27.18",  
  "Fashion accessories_27.38",  
  "Fashion accessories_29.42",  
  "Fashion accessories_29.56",  
  "Fashion accessories_30.14",  
  "Fashion accessories_30.37",  
  "Fashion accessories_30.61",  
  "Fashion accessories_30.62",  
  "Fashion accessories_31.73",  
  "Fashion accessories_31.9",  
  "Fashion accessories_32.62",  
  "Fashion accessories_33.52",  
  "Fashion accessories_33.63",  
  "Fashion accessories_34.7",  
  "Fashion accessories_35.19",  
  "Fashion accessories_36.51",  
  "Fashion accessories_36.85",  
  "Fashion accessories_37.15",  
  "Fashion accessories_37.55",  
  "Fashion accessories_37.95"  
$\}$

**Parameters:**  
For each $i \in \mathcal{F}$:
- $A_i$ = Revenue per unit of product $i$ (from "Revenue" column)
- $d_i$ = Demand for product $i$ (from "Demand" column)
- $I_i$ = Initial Inventory for product $i$ (from "Initial Inventory" column)

**Parameter Table (source order):**

| Product Name                   | $A_i$   | $d_i$ | $I_i$ |
|------------------------------- |--------:|------:|------:|
| Fashion accessories_10.18      | 10.18   | 12    | 80    |
| Fashion accessories_12.09      | 12.09   | 2     | 10    |
| Fashion accessories_12.19      | 12.19   | 10    | 80    |
| Fashion accessories_12.54      | 12.54   | 2     | 10    |
| Fashion accessories_12.78      | 12.78   | 2     | 10    |
| Fashion accessories_14.48      | 14.48   | 5     | 40    |
| Fashion accessories_15.43      | 15.43   | 2     | 10    |
| Fashion accessories_15.5       | 15.5    | 2     | 10    |
| Fashion accessories_15.62      | 15.62   | 11    | 80    |
| Fashion accessories_16.28      | 16.28   | 2     | 10    |
| Fashion accessories_16.45      | 16.45   | 6     | 40    |
| Fashion accessories_17.48      | 17.48   | 9     | 60    |
| Fashion accessories_17.49      | 17.49   | 14    | 100   |
| Fashion accessories_17.87      | 17.87   | 6     | 40    |
| Fashion accessories_17.94      | 17.94   | 8     | 50    |
| Fashion accessories_18.08      | 18.08   | 6     | 40    |
| Fashion accessories_19.66      | 19.66   | 14    | 100   |
| Fashion accessories_19.7       | 19.7    | 2     | 10    |
| Fashion accessories_19.77      | 19.77   | 14    | 100   |
| Fashion accessories_20.01      | 20.01   | 11    | 90    |
| Fashion accessories_21.32      | 21.32   | 2     | 10    |
| Fashion accessories_21.48      | 21.48   | 3     | 20    |
| Fashion accessories_21.94      | 21.94   | 8     | 50    |
| Fashion accessories_22.32      | 22.32   | 10    | 80    |
| Fashion accessories_22.51      | 22.51   | 10    | 70    |
| Fashion accessories_23.82      | 23.82   | 7     | 50    |
| Fashion accessories_25.42      | 25.42   | 11    | 80    |
| Fashion accessories_25.56      | 25.56   | 11    | 70    |
| Fashion accessories_27.02      | 27.02   | 5     | 30    |
| Fashion accessories_27.18      | 27.18   | 3     | 20    |
| Fashion accessories_27.38      | 27.38   | 8     | 60    |
| Fashion accessories_29.42      | 29.42   | 13    | 100   |
| Fashion accessories_29.56      | 29.56   | 7     | 50    |
| Fashion accessories_30.14      | 30.14   | 14    | 100   |
| Fashion accessories_30.37      | 30.37   | 4     | 30    |
| Fashion accessories_30.61      | 30.61   | 2     | 10    |
| Fashion accessories_30.62      | 30.62   | 2     | 10    |
| Fashion accessories_31.73      | 31.73   | 14    | 90    |
| Fashion accessories_31.9       | 31.9    | 2     | 10    |
| Fashion accessories_32.62      | 32.62   | 6     | 40    |
| Fashion accessories_33.52      | 33.52   | 2     | 10    |
| Fashion accessories_33.63      | 33.63   | 2     | 10    |
| Fashion accessories_34.7       | 34.7    | 3     | 20    |
| Fashion accessories_35.19      | 35.19   | 14    | 100   |
| Fashion accessories_36.51      | 36.51   | 12    | 90    |
| Fashion accessories_36.85      | 36.85   | 7     | 50    |
| Fashion accessories_37.15      | 37.15   | 6     | 40    |
| Fashion accessories_37.55      | 37.55   | 13    | 100   |
| Fashion accessories_37.95      | 37.95   | 14    | 100   |

**Decision Variables:**  
For each $i \in \mathcal{F}$:
- $x_i$ = number of units of product $i$ to fulfill  
  $x_i \in \mathbb{Z}_+$ (non-negative integer)

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in \mathcal{F}} A_i \cdot x_i
$$

**Constraints:**  
For each $i \in \mathcal{F}$:
- Demand and inventory bounds:
$$
0 \leq x_i \leq \min\{d_i, I_i\}
$$

**Full Model:**

**Parameters:**  
- $\mathcal{F}$: as listed above (49 products, source order)
- $A_i$, $d_i$, $I_i$: as in the table above

**Variables:**  
- $x_i \in \mathbb{Z}_+$, for all $i \in \mathcal{F}$

**Objective:**  
$$
\max \sum_{i \in \mathcal{F}} A_i \cdot x_i
$$

**Subject to:**  
For all $i \in \mathcal{F}$,
$$
0 \leq x_i \leq \min\{d_i, I_i\}
$$

**Retrieved Information:**  
- Product Names (source order):  
  "Fashion accessories_10.18", "Fashion accessories_12.09", "Fashion accessories_12.19", "Fashion accessories_12.54", "Fashion accessories_12.78", "Fashion accessories_14.48", "Fashion accessories_15.43", "Fashion accessories_15.5", "Fashion accessories_15.62", "Fashion accessories_16.28", "Fashion accessories_16.45", "Fashion accessories_17.48", "Fashion accessories_17.49", "Fashion accessories_17.87", "Fashion accessories_17.94", "Fashion accessories_18.08", "Fashion accessories_19.66", "Fashion accessories_19.7", "Fashion accessories_19.77", "Fashion accessories_20.01", "Fashion accessories_21.32", "Fashion accessories_21.48", "Fashion accessories_21.94", "Fashion accessories_22.32", "Fashion accessories_22.51", "Fashion accessories_23.82", "Fashion accessories_25.42", "Fashion accessories_25.56", "Fashion accessories_27.02", "Fashion accessories_27.18", "Fashion accessories_27.38", "Fashion accessories_29.42", "Fashion accessories_29.56", "Fashion accessories_30.14", "Fashion accessories_30.37", "Fashion accessories_30.61", "Fashion accessories_30.62", "Fashion accessories_31.73", "Fashion accessories_31.9", "Fashion accessories_32.62", "Fashion accessories_33.52", "Fashion accessories_33.63", "Fashion accessories_34.7", "Fashion accessories_35.19", "Fashion accessories_36.51", "Fashion accessories_36.85", "Fashion accessories_37.15", "Fashion accessories_37.55", "Fashion accessories_37.95"
- Revenue vector $A_i$:  
  [10.18, 12.09, 12.19, 12.54, 12.78, 14.48, 15.43, 15.5, 15.62, 16.28, 16.45, 17.48, 17.49, 17.87, 17.94, 18.08, 19.66, 19.7, 19.77, 20.01, 21.32, 21.48, 21.94, 22.32, 22.51, 23.82, 25.42, 25.56, 27.02, 27.18, 27.38, 29.42, 29.56, 30.14, 30.37, 30.61, 30.62, 31.73, 31.9, 32.62, 33.52, 33.63, 34.7, 35.19, 36.51, 36.85, 37.15, 37.55, 37.95]
- Demand vector $d_i$:  
  [12, 2, 10, 2, 2, 5, 2, 2, 11, 2, 6, 9, 14, 6, 8, 6, 14, 2, 14, 11, 2, 3, 8, 10, 10, 7, 11, 11, 5, 3, 8, 13, 7, 14, 4, 2, 2, 14, 2, 6, 2, 2, 3, 14, 12, 7, 6, 13, 14]
- Initial Inventory vector $I_i$:  
  [80, 10, 80, 10, 10, 40, 10, 10, 80, 10, 40, 60, 100, 40, 50, 40, 100, 10, 100, 90, 10, 20, 50, 80, 70, 50, 80, 70, 30, 20, 60, 100, 50, 100, 30, 10, 10, 90, 10, 40, 10, 10, 20, 100, 90, 50, 40, 100, 100]

**Variable bounds:**  
For each $i$ (in source order):
- $0 \leq x_i \leq \min\{d_i, I_i\}$

**Variable domains:**  
- $x_i \in \mathbb{Z}_+$ for all $i \in \mathcal{F}$

**End of Model.**