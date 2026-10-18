Let $x_i$ denote the number of units of product $i$ to fulfill (allocate to demand), where $i$ indexes the following products in the order given:

1. Beauty - 25
2. Beauty - 30
3. Beauty - 300
4. Beauty - 50
5. Beauty - 500
6. Clothing - 25
7. Clothing - 30
8. Clothing - 300
9. Clothing - 50
10. Clothing - 500
11. Electronics - 25
12. Electronics - 30
13. Electronics - 300
14. Electronics - 50
15. Electronics - 500
16. Home Goods - 25
17. Home Goods - 30
18. Home Goods - 50
19. Home Goods - 300
20. Home Goods - 500
21. Sports - 25
22. Sports - 30
23. Sports - 50
24. Sports - 300
25. Sports - 500
26. Furniture - 25
27. Furniture - 30
28. Furniture - 50
29. Furniture - 300
30. Furniture - 500
31. Toys - 25
32. Toys - 30
33. Toys - 50
34. Toys - 300
35. Toys - 500

Parameters for each product $i$ (in the same order):

- $r_i$: Revenue per unit
- $d_i$: Demand
- $s_i$: Initial Inventory

The data for each product $i$ is as follows:

| $i$ | Product Name           | $r_i$ | $d_i$ | $s_i$ |
|-----|-----------------------|-------|-------|-------|
| 1   | Beauty - 25           | 25    | 240   | 1570  |
| 2   | Beauty - 30           | 30    | 202   | 1330  |
| 3   | Beauty - 300          | 300   | 216   | 1420  |
| 4   | Beauty - 50           | 50    | 263   | 1700  |
| 5   | Beauty - 500          | 500   | 256   | 1690  |
| 6   | Clothing - 25         | 25    | 281   | 1840  |
| 7   | Clothing - 30         | 30    | 261   | 1710  |
| 8   | Clothing - 300        | 300   | 295   | 1930  |
| 9   | Clothing - 50         | 50    | 290   | 1890  |
| 10  | Clothing - 500        | 500   | 244   | 1570  |
| 11  | Electronics - 25      | 25    | 273   | 1810  |
| 12  | Electronics - 30      | 30    | 220   | 1410  |
| 13  | Electronics - 300     | 300   | 286   | 1830  |
| 14  | Electronics - 50      | 50    | 268   | 1750  |
| 15  | Electronics - 500     | 500   | 262   | 1690  |
| 16  | Home Goods - 25       | 25    | 255   | 1660  |
| 17  | Home Goods - 30       | 30    | 218   | 1417  |
| 18  | Home Goods - 50       | 50    | 278   | 1807  |
| 19  | Home Goods - 300      | 300   | 195   | 1268  |
| 20  | Home Goods - 500      | 500   | 248   | 1612  |
| 21  | Sports - 25           | 25    | 269   | 1749  |
| 22  | Sports - 30           | 30    | 227   | 1476  |
| 23  | Sports - 50           | 50    | 285   | 1853  |
| 24  | Sports - 300          | 300   | 208   | 1352  |
| 25  | Sports - 500          | 500   | 259   | 1684  |
| 26  | Furniture - 25        | 25    | 242   | 1573  |
| 27  | Furniture - 30        | 30    | 213   | 1385  |
| 28  | Furniture - 50        | 50    | 272   | 1768  |
| 29  | Furniture - 300       | 300   | 224   | 1456  |
| 30  | Furniture - 500       | 500   | 251   | 1632  |
| 31  | Toys - 25             | 25    | 257   | 1671  |
| 32  | Toys - 30             | 30    | 235   | 1528  |
| 33  | Toys - 50             | 50    | 291   | 1892  |
| 34  | Toys - 300            | 300   | 199   | 1294  |
| 35  | Toys - 500            | 500   | 264   | 1716  |

The mathematical model is:

Maximize total revenue:
$$
\max \sum_{i=1}^{35} r_i x_i
$$

Subject to, for each $i = 1, \ldots, 35$:
\[
\begin{align*}
& x_i \leq d_i \\
& x_i \leq s_i \\
& x_i \geq 0 \\
& x_i \in \mathbb{Z}
\end{align*}
\]

Where:
- $x_i$ = number of units of product $i$ to fulfill (allocate to demand)
- $r_i$ = revenue per unit for product $i$ (see table above)
- $d_i$ = demand for product $i$ (see table above)
- $s_i$ = initial inventory for product $i$ (see table above)

All variables $x_i$ are nonnegative integers.