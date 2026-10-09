Let $I$ be the set of all dairy products, indexed by their Full_Product_Name as given in the data. For each product $i \in I$:

- Let $x_i$ = number of units of product $i$ to be fulfilled (decision variable, nonnegative integer)
- Let $r_i$ = Revenue per unit of product $i$ (from 'Revenue' column)
- Let $d_i$ = Demand for product $i$ (from 'Demand' column)
- Let $s_i$ = Initial Inventory for product $i$ (from 'Initial Inventory' column)

The complete mathematical model is:

$$
\begin{align*}
\text{Maximize} \quad & \sum_{i \in I} r_i \, x_i \\[2ex]
\text{subject to} \quad
& 0 \leq x_i \leq \min\{d_i,\, s_i\} \qquad \forall i \in I \\
& x_i \in \mathbb{Z}_{\geq 0} \qquad\qquad\qquad\quad \forall i \in I
\end{align*}
$$

Where the parameters for each product $i$ are as follows (in source order):

| Full_Product_Name                | $r_i$ (Revenue) | $d_i$ (Demand) | $s_i$ (Initial Inventory) |
|----------------------------------|-----------------|----------------|--------------------------|
| Butter_Amul                      | 96.86           | 34102          | 29862                    |
| Butter_Mother Dairy              | 48.01           | 36579          | 29898                    |
| Butter_Parag Milk Foods          | 8.83            | 36086          | 25208                    |
| Butter_Warana                    | 92.96           | 41254          | 30816                    |
| Buttermilk_Amul                  | 40.75           | 29876          | 19925                    |
| Buttermilk_Mother Dairy          | 83.07           | 41229          | 26482                    |
| Buttermilk_Raj                   | 15.64           | 35354          | 30865                    |
| Buttermilk_Sudha                 | 56.57           | 29649          | 33517                    |
| Cheese_Amul                      | 100.74          | 38558          | 30929                    |
| Cheese_Britannia Industries      | 28.92           | 28603          | 21405                    |
| Cheese_Dynamix Dairies           | 32.66           | 35962          | 25953                    |
| Cheese_Passion Cheese            | 58.09           | 36961          | 23825                    |
| Curd_Amul                        | 30.27           | 39436          | 31687                    |
| Curd_Mother Dairy                | 84.57           | 43522          | 33377                    |
| Curd_Raj                         | 84.75           | 38128          | 34914                    |
| Curd_Sudha                       | 76.37           | 42341          | 33547                    |
| Ghee_Amul                        | 41.49           | 30345          | 23120                    |
| Ghee_Mother Dairy                | 52.79           | 35420          | 24667                    |
| Ghee_Raj                         | 48.13           | 34100          | 25395                    |
| Ghee_Sudha                       | 95.09           | 33007          | 24676                    |
| Ice Cream_Amul                   | 54.41           | 37894          | 26707                    |
| Ice Cream_Dodla Dairy            | 82.24           | 29840          | 26722                    |
| Ice Cream_Mother Dairy           | 94.32           | 38762          | 25809                    |
| Ice Cream_Palle2patnam           | 83.73           | 34674          | 24391                    |
| Lassi_Amul                       | 74.45           | 42972          | 30728                    |
| Lassi_Mother Dairy               | 49.4            | 33894          | 28628                    |
| Lassi_Raj                        | 93.93           | 45762          | 30568                    |
| Lassi_Sudha                      | 88.05           | 29503          | 23461                    |
| Milk_Amul                        | 39.24           | 34761          | 21398                    |
| Milk_Mother Dairy                | 8.69            | 40548          | 33619                    |
| Milk_Raj                         | 65.53           | 43012          | 26355                    |
| Milk_Sudha                       | 42.34           | 29180          | 23815                    |
| Paneer_Amul                      | 81.76           | 33498          | 20787                    |
| Paneer_Mother Dairy              | 29.09           | 34848          | 29342                    |
| Paneer_Raj                       | 87.3            | 40347          | 23556                    |
| Paneer_Sudha                     | 66.7            | 37188          | 28753                    |
| Yogurt_Amul                      | 89.32           | 34347          | 24404                    |
| Yogurt_Dodla Dairy               | 33.81           | 37181          | 26829                    |
| Yogurt_Mother Dairy              | 25.29           | 36644          | 25562                    |
| Yogurt_Palle2patnam              | 84.9            | 34303          | 28695                    |

Where $x_i$ is the number of units of product $i$ fulfilled, for each $i$ in the above table.