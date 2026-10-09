Let $x_{i}$ be the production quantity of product $i$ ($i=1,2,3$ for I, II, III).  
Let $y_{k,i}$ be the quantity of product $i$ processed on equipment $k$ (for eligible $k,i$ pairs).

#### Parameters (from 43.csv, preserving source order)

- Equipment: $A1$, $A2$, $A3$, $B1$, $B2$, $B3$, $B4$
- Products: I ($i=1$), II ($i=2$), III ($i=3$)
- Processing time per unit (minutes/unit) for each equipment and product (blank means not eligible):

| Equipment | I | II | III | Available Time | Equipment Cost at Full Load |
|-----------|---|----|------|---------------|----------------------------|
| A1        | 5 | 10 |      | 6000          | 300                        |
| A2        | 7 | 9  | 12   | 10000         | 321                        |
| A3        | 6 | 11 | 2    | 8000          | 203                        |
| B1        | 6 | 8  |      | 4000          | 250                        |
| B2        | 4 |    | 11   | 7000          | 783                        |
| B3        | 7 |    |      | 4000          | 200                        |
| B4        | 3 | 5  | 8    | 5000          | 300                        |

- Raw material cost (yuan/unit): I: 0.25, II: 0.35, III: 0.5
- Unit price (yuan/unit): I: 1.25, II: 2, III: 2.8

#### Eligibility (from user description and table):

- Procedure A: A1, A2, A3
  - I: A1, A2, A3
  - II: A1, A2, A3
  - III: A2, A3
- Procedure B: B1, B2, B3, B4
  - I: B1, B2, B3, B4
  - II: B1, B4
  - III: B2, B4

#### Decision Variables

- $x_1, x_2, x_3 \geq 0$: production quantity of product I, II, III
- $y_{k,i} \geq 0$: quantity of product $i$ processed on equipment $k$ (only for eligible $k,i$)

#### Objective Function

Maximize total profit:
\[
\max \left[
(1.25-0.25)x_1 + (2-0.35)x_2 + (2.8-0.5)x_3
- \sum_{k} \text{Equipment Cost}_k \cdot \frac{\sum_{i} t_{k,i} y_{k,i}}{\text{Available Time}_k}
\right]
\]
where $t_{k,i}$ is the processing time per unit for product $i$ on equipment $k$ (from table, blank means not eligible, so $y_{k,i}=0$).

#### Constraints

1. **Production-Procedure Assignment:**  
   For each product, the sum of quantities processed on eligible equipment for each procedure equals the production quantity:
   - Procedure A:
     - $x_1 = y_{A1,1} + y_{A2,1} + y_{A3,1}$
     - $x_2 = y_{A1,2} + y_{A2,2} + y_{A3,2}$
     - $x_3 = y_{A2,3} + y_{A3,3}$
   - Procedure B:
     - $x_1 = y_{B1,1} + y_{B2,1} + y_{B3,1} + y_{B4,1}$
     - $x_2 = y_{B1,2} + y_{B4,2}$
     - $x_3 = y_{B2,3} + y_{B4,3}$

2. **Equipment Capacity:**  
   For each equipment $k$, total processing time cannot exceed available time:
   \[
   \sum_{i} t_{k,i} y_{k,i} \leq \text{Available Time}_k
   \]
   (sum only over eligible $i$ for each $k$)

3. **Non-negativity:**  
   $x_i \geq 0$, $y_{k,i} \geq 0$ for all eligible $k,i$.

#### Full Model (with all coefficients):

Let $t_{k,i}$ be the processing time per unit for product $i$ on equipment $k$ (from table below).

**Parameters:**

- $t_{A1,1}=5$, $t_{A1,2}=10$
- $t_{A2,1}=7$, $t_{A2,2}=9$, $t_{A2,3}=12$
- $t_{A3,1}=6$, $t_{A3,2}=11$, $t_{A3,3}=2$
- $t_{B1,1}=6$, $t_{B1,2}=8$
- $t_{B2,1}=4$, $t_{B2,3}=11$
- $t_{B3,1}=7$
- $t_{B4,1}=3$, $t_{B4,2}=5$, $t_{B4,3}=8$

- Equipment costs at full load: $c_{A1}=300$, $c_{A2}=321$, $c_{A3}=203$, $c_{B1}=250$, $c_{B2}=783$, $c_{B3}=200$, $c_{B4}=300$
- Available times: $T_{A1}=6000$, $T_{A2}=10000$, $T_{A3}=8000$, $T_{B1}=4000$, $T_{B2}=7000$, $T_{B3}=4000$, $T_{B4}=5000$

**Variables:**

- $x_1, x_2, x_3 \geq 0$
- $y_{A1,1}, y_{A1,2} \geq 0$
- $y_{A2,1}, y_{A2,2}, y_{A2,3} \geq 0$
- $y_{A3,1}, y_{A3,2}, y_{A3,3} \geq 0$
- $y_{B1,1}, y_{B1,2} \geq 0$
- $y_{B2,1}, y_{B2,3} \geq 0$
- $y_{B3,1} \geq 0$
- $y_{B4,1}, y_{B4,2}, y_{B4,3} \geq 0$

**Objective:**
\[
\max \Bigg[
(1.25-0.25)x_1 + (2-0.35)x_2 + (2.8-0.5)x_3
- 300 \cdot \frac{5y_{A1,1} + 10y_{A1,2}}{6000}
- 321 \cdot \frac{7y_{A2,1} + 9y_{A2,2} + 12y_{A2,3}}{10000}
- 203 \cdot \frac{6y_{A3,1} + 11y_{A3,2} + 2y_{A3,3}}{8000}
- 250 \cdot \frac{6y_{B1,1} + 8y_{B1,2}}{4000}
- 783 \cdot \frac{4y_{B2,1} + 11y_{B2,3}}{7000}
- 200 \cdot \frac{7y_{B3,1}}{4000}
- 300 \cdot \frac{3y_{B4,1} + 5y_{B4,2} + 8y_{B4,3}}{5000}
\Bigg]
\]

**Subject to:**

- $x_1 = y_{A1,1} + y_{A2,1} + y_{A3,1}$
- $x_2 = y_{A1,2} + y_{A2,2} + y_{A3,2}$
- $x_3 = y_{A2,3} + y_{A3,3}$

- $x_1 = y_{B1,1} + y_{B2,1} + y_{B3,1} + y_{B4,1}$
- $x_2 = y_{B1,2} + y_{B4,2}$
- $x_3 = y_{B2,3} + y_{B4,3}$

- $5y_{A1,1} + 10y_{A1,2} \leq 6000$
- $7y_{A2,1} + 9y_{A2,2} + 12y_{A2,3} \leq 10000$
- $6y_{A3,1} + 11y_{A3,2} + 2y_{A3,3} \leq 8000$
- $6y_{B1,1} + 8y_{B1,2} \leq 4000$
- $4y_{B2,1} + 11y_{B2,3} \leq 7000$
- $7y_{B3,1} \leq 4000$
- $3y_{B4,1} + 5y_{B4,2} + 8y_{B4,3} \leq 5000$

- All variables $\geq 0$

All coefficients and eligibility are preserved from the source data and user description.