Let $x_i$ be the number of units of product $i$ to be fulfilled, where $i$ indexes the following products in the order given:

1. Butter_Amul
2. Butter_Mother Dairy
3. Butter_Parag Milk Foods
4. Butter_Warana
5. Buttermilk_Amul
6. Buttermilk_Mother Dairy
7. Buttermilk_Raj
8. Buttermilk_Sudha
9. Cheese_Amul
10. Cheese_Britannia Industries
11. Cheese_Dynamix Dairies
12. Cheese_Passion Cheese
13. Curd_Amul
14. Curd_Mother Dairy
15. Curd_Raj
16. Curd_Sudha
17. Ghee_Amul
18. Ghee_Mother Dairy
19. Ghee_Raj
20. Ghee_Sudha
21. Ice Cream_Amul
22. Ice Cream_Dodla Dairy
23. Ice Cream_Mother Dairy
24. Ice Cream_Palle2patnam
25. Lassi_Amul
26. Lassi_Mother Dairy
27. Lassi_Raj
28. Lassi_Sudha
29. Milk_Amul
30. Milk_Mother Dairy
31. Milk_Raj
32. Milk_Sudha
33. Paneer_Amul
34. Paneer_Mother Dairy
35. Paneer_Raj
36. Paneer_Sudha
37. Yogurt_Amul
38. Yogurt_Dodla Dairy
39. Yogurt_Mother Dairy
40. Yogurt_Palle2patnam

Parameters for each product $i$:

- $r_i$: Revenue per unit
- $d_i$: Demand
- $s_i$: Initial Inventory

All data is as follows (in source order):

| $i$ | Full_Product_Name                | $r_i$  | $d_i$  | $s_i$  |
|-----|----------------------------------|--------|--------|--------|
| 1   | Butter_Amul                      | 96.86  | 34102  | 29862  |
| 2   | Butter_Mother Dairy              | 48.01  | 36579  | 29898  |
| 3   | Butter_Parag Milk Foods          | 8.83   | 36086  | 25208  |
| 4   | Butter_Warana                    | 92.96  | 41254  | 30816  |
| 5   | Buttermilk_Amul                  | 40.75  | 29876  | 19925  |
| 6   | Buttermilk_Mother Dairy          | 83.07  | 41229  | 26482  |
| 7   | Buttermilk_Raj                   | 15.64  | 35354  | 30865  |
| 8   | Buttermilk_Sudha                 | 56.57  | 29649  | 33517  |
| 9   | Cheese_Amul                      | 100.74 | 38558  | 30929  |
| 10  | Cheese_Britannia Industries      | 28.92  | 28603  | 21405  |
| 11  | Cheese_Dynamix Dairies           | 32.66  | 35962  | 25953  |
| 12  | Cheese_Passion Cheese            | 58.09  | 36961  | 23825  |
| 13  | Curd_Amul                        | 30.27  | 39436  | 31687  |
| 14  | Curd_Mother Dairy                | 84.57  | 43522  | 33377  |
| 15  | Curd_Raj                         | 84.75  | 38128  | 34914  |
| 16  | Curd_Sudha                       | 76.37  | 42341  | 33547  |
| 17  | Ghee_Amul                        | 41.49  | 30345  | 23120  |
| 18  | Ghee_Mother Dairy                | 52.79  | 35420  | 24667  |
| 19  | Ghee_Raj                         | 48.13  | 34100  | 25395  |
| 20  | Ghee_Sudha                       | 95.09  | 33007  | 24676  |
| 21  | Ice Cream_Amul                   | 54.41  | 37894  | 26707  |
| 22  | Ice Cream_Dodla Dairy            | 82.24  | 29840  | 26722  |
| 23  | Ice Cream_Mother Dairy           | 94.32  | 38762  | 25809  |
| 24  | Ice Cream_Palle2patnam           | 83.73  | 34674  | 24391  |
| 25  | Lassi_Amul                       | 74.45  | 42972  | 30728  |
| 26  | Lassi_Mother Dairy               | 49.4   | 33894  | 28628  |
| 27  | Lassi_Raj                        | 93.93  | 45762  | 30568  |
| 28  | Lassi_Sudha                      | 88.05  | 29503  | 23461  |
| 29  | Milk_Amul                        | 39.24  | 34761  | 21398  |
| 30  | Milk_Mother Dairy                | 8.69   | 40548  | 33619  |
| 31  | Milk_Raj                         | 65.53  | 43012  | 26355  |
| 32  | Milk_Sudha                       | 42.34  | 29180  | 23815  |
| 33  | Paneer_Amul                      | 81.76  | 33498  | 20787  |
| 34  | Paneer_Mother Dairy              | 29.09  | 34848  | 29342  |
| 35  | Paneer_Raj                       | 87.3   | 40347  | 23556  |
| 36  | Paneer_Sudha                     | 66.7   | 37188  | 28753  |
| 37  | Yogurt_Amul                      | 89.32  | 34347  | 24404  |
| 38  | Yogurt_Dodla Dairy               | 33.81  | 37181  | 26829  |
| 39  | Yogurt_Mother Dairy              | 25.29  | 36644  | 25562  |
| 40  | Yogurt_Palle2patnam              | 84.9   | 34303  | 28695  |

The mathematical model is:

**Objective:**
\[
\max \sum_{i=1}^{40} r_i x_i
\]

**Subject to:**

- Demand fulfillment and inventory constraints for each product $i$:
  \[
  0 \leq x_i \leq \min\{d_i, s_i\} \qquad \forall i = 1, \ldots, 40
  \]
  (i.e., cannot fulfill more than demand or available inventory)

- Integrality:
  \[
  x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1, \ldots, 40
  \]

**Where:**

- $x_i$ = number of units of product $i$ fulfilled
- $r_i$ = revenue per unit of product $i$ (see table above)
- $d_i$ = demand for product $i$ (see table above)
- $s_i$ = initial inventory for product $i$ (see table above)

All indices, coefficients, and bounds are as given in the table above, in the original source order.