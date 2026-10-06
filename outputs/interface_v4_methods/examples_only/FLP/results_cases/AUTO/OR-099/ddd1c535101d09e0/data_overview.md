Here are the transportation costs \( c_{ij} \) from each warehouse \( i \) to each store \( j \), as provided in the TransportationCost.csv. The matrix is organized with warehouses as both row and column headers, matching the original source orientation and shape. Each row corresponds to a warehouse (W1 to W11), and each column corresponds to a store (W1 to W11). The values represent the transportation cost from warehouse \( i \) (row) to store \( j \) (column):

| Source Row (Warehouse) | W1 | W2 | W3 | W4 | W5 | W6 | W7 | W8 | W9 | W10 | W11 |
|------------------------|----|----|----|----|----|----|----|----|----|-----|-----|
| W1                     | 12 | 11 | 14 | 15 | 17 | 13 | 12 | 16 | 16 | 14  | 15  |
| W2                     | 17 | 19 | 15 | 20 | 18 | 14 | 17 | 15 | 13 | 15  | 16  |
| W3                     | 13 | 14 | 12 | 14 | 16 | 15 | 11 | 14 | 16 | 18  | 17  |
| W4                     | 18 | 16 | 17 | 13 | 18 | 17 | 14 | 19 | 16 | 13  | 18  |
| W5                     | 10 | 13 | 12 | 19 | 15 | 11 | 12 | 14 | 12 | 15  | 17  |
| W6                     | 15 | 12 | 14 | 16 | 13 | 17 | 16 | 16 | 14 | 18  | 19  |
| W7                     | 14 | 13 | 15 | 17 | 12 | 13 | 14 | 15 | 12 | 16  | 14  |
| W8                     | 19 | 16 | 18 | 20 | 17 | 19 | 16 | 18 | 15 | 15  | 18  |
| W9                     | 17 | 18 | 12 | 14 | 16 | 15 | 14 | 17 | 21 | 15  | 18  |
| W10                    | 14 | 13 | 15 | 17 | 16 | 18 | 14 | 19 | 15 | 17  | 19  |
| W11                    | 15 | 13 | 16 | 17 | 11 | 13 | 14 | 15 | 19 | 21  | 13  |

- Each row is indexed by warehouse ID (W1 to W11, corresponding to warehouse \( i \)).
- Each column is indexed by store ID (W1 to W11, corresponding to store \( j \)).
- The value at (row \( i \), column \( j \)) is the transportation cost \( c_{ij} \) from warehouse \( i \) to store \( j \).
- The matrix orientation and axis labels are preserved as in the source.