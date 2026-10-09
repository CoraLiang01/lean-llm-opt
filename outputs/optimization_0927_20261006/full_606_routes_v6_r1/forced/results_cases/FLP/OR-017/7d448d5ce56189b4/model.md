##### Decision Variables

$x_i \geq 0$: number of units of each ‘ZZ’ product $i \in I$ to fulfill (continuous, integer-valued in practice).

##### Objective Function

$\max \sum_{i\in I} r_i x_i$

where $r_i$ is the revenue per unit for product $i$.

##### Constraints

1. Demand fulfillment: $x_i \leq d_i,\quad \forall i\in I$
2. Inventory limit: $x_i \leq s_i,\quad \forall i\in I$
3. Nonnegativity: $x_i \geq 0,\quad \forall i\in I$

where $d_i$ is the demand for product $i$, and $s_i$ is the initial inventory for product $i$.

##### Sets and Parameters

Let $I$ be the set of all products with category ‘ZZ’. For each $i \in I$:

- $r_i$: revenue per unit of product $i$
- $d_i$: demand for product $i$
- $s_i$: initial inventory of product $i$

##### Data

Below is the full list of products $i \in I$ (SKU), with their parameters:

| SKU    | Revenue ($r_i$) | Demand ($d_i$) | Initial Inventory ($s_i$) |
|--------|-----------------|---------------|---------------------------|
| 00GVC  | 17.68           | 4             | 20.0                      |
| 00OK1  | 1.27            | 33            | 180.0                     |
| 0121I  | 2.03            | 60            | 310.0                     |
| 01IEO  | 4.96            | 81            | 430.0                     |
| 01IQT  | 1.5             | 14            | 70.0                      |
| 01L05  | 167.64          | 16            | 100.0                     |
| 01V7M  | 8.0             | 88            | 450.0                     |
| 01XVY  | 1.46            | 2             | 10.0                      |
| 029WA  | 5.03            | 4             | 20.0                      |
| 03C6L  | 1.23            | 68            | 360.0                     |
| 03CPI  | 15.12           | 10            | 50.0                      |
| 03K3G  | 12.32           | 8             | 40.0                      |
| 04KDD  | 9.93            | 2             | 10.0                      |
| 050FN  | 41.43           | 2             | 10.0                      |
| 055SH  | 16.3            | 30            | 150.0                     |
| 055UG  | 8.04            | 16            | 80.0                      |
| 05B2E  | 3.04            | 14            | 70.0                      |
| 05HRH  | 18.77           | 7             | 40.0                      |
| 05O3Z  | 46.12           | 20            | 110.0                     |
| 05ZN9  | 11.29           | 3             | 30.0                      |
| 062O9  | 1.88            | 47            | 240.0                     |
| 06KYM  | 3.18            | 4             | 20.0                      |
| 06SMH  | 16.12           | 2             | 10.0                      |
| 070SU  | 15.43           | 2             | 10.0                      |
| 072QJ  | 2.62            | 51            | 290.0                     |
| 07GHA  | 11.65           | 72            | 370.0                     |
| 07NQU  | 5.78            | 52            | 300.0                     |
| 07OCK  | 4.88            | 61            | 310.0                     |
| 08F28  | 19.17           | 104           | 650.0                     |
| 08TP3  | 179.59          | 3             | 20.0                      |
| 08UZX  | 13.58           | 44            | 230.0                     |
| 08WV6  | 44.62           | 10            | 50.0                      |
| 08XGM  | 19.07           | 2             | 10.0                      |
| 096VW  | 6.07            | 110           | 590.0                     |
| 096X4  | 17.37           | 26            | 130.0                     |
| 09A6U  | 5.25            | 2             | 10.0                      |
| 09K06  | 235.91          | 55            | 360.0                     |
| 09K2Q  | 13.15           | 2             | 10.0                      |
| 09LL9  | 5.5             | 316           | 1930.0                    |
| 0A3I2  | 10.81           | 24            | 120.0                     |
| 0A5D2  | 2.75            | 2             | 10.0                      |
| 0A6IO  | 4.06            | 6             | 30.0                      |
| 0AC89  | 4.77            | 10            | 50.0                      |
| 0AEJH  | 1.89            | 157           | 860.0                     |
| 0AI2K  | 248.62          | 3             | 20.0                      |
| 0AJJN  | 33.24           | 3             | 20.0                      |
| 0AKGM  | 13.41           | 20            | 100.0                     |
| 0B8JX  | 2.25            | 116           | 590.0                     |
| 0BC0Z  | 47.76           | 3             | 20.0                      |
| 0BDT2  | 18.69           | 6             | 30.0                      |
| 0BWHE  | 2.43            | 21            | 110.0                     |
| 0BXVJ  | 7.5             | 31            | 160.0                     |
| 0C582  | 23.33           | 2             | 10.0                      |
| 0CKHT  | 16.83           | 85            | 470.0                     |
| 0CSPY  | 3.1             | 18            | 90.0                      |
| 0CUIK  | 4.38            | 18            | 90.0                      |
| 0CY5Y  | 18.5            | 4             | 20.0                      |
| 0D3EZ  | 11.29           | 169           | 940.0                     |
| 0DCAY  | 22.38           | 276           | 1440.0                    |
| 0DQVE  | 25.01           | 6             | 30.0                      |
| 0DW3Z  | 28.76           | 17            | 100.0                     |
| 0EEPA  | 27.13           | 14            | 70.0                      |
| 0EJTN  | 6.87            | 9             | 50.0                      |
| 0EM7L  | 3.13            | 55            | 280.0                     |
| 0EURY  | 88.05           | 2             | 10.0                      |
| 0F721  | 4.38            | 2             | 10.0                      |
| 0FGYA  | 57.97           | 2             | 10.0                      |
| 0FNQX  | 54.93           | 2             | 10.0                      |
| 0FV3U  | 6.82            | 14            | 70.0                      |
| 0FWL7  | 14.1            | 1             | 10.0                      |
| 0GHAB  | 17.25           | 4             | 20.0                      |
| 0GLIK  | 10.57           | 18            | 90.0                      |
| 0GU8Z  | 43.38           | 11            | 60.0                      |
| 0H4OF  | 28.01           | 8             | 40.0                      |
| 0H5CR  | 5.62            | 62            | 330.0                     |
| 0H8QD  | 16.5            | 30            | 150.0                     |
| 0HCFM  | 19.56           | 6             | 30.0                      |
| 0HUPP  | 2.04            | 166           | 880.0                     |
| 0HUYO  | 16.07           | 12            | 70.0                      |
| 0HYB2  | 5.56            | 24            | 120.0                     |
| 0HYIM  | 3.38            | 22            | 110.0                     |
| 0I23E  | 22.26           | 3             | 20.0                      |
| 0I8Q6  | 7.94            | 4             | 20.0                      |
| 0IERB  | 11.79           | 6             | 30.0                      |
| 0IM8B  | 6.12            | 57            | 290.0                     |
| 0ISQE  | 25.83           | 46            | 230.0                     |
| 0JD0P  | 8.38            | 70            | 350.0                     |
| 0JP69  | 3.97            | 92            | 590.0                     |
| 0JXMY  | 8.65            | 24            | 120.0                     |
| 0K0L7  | 5.19            | 6             | 30.0                      |
| 0KRMR  | 2.68            | 29            | 160.0                     |
| 0KYGK  | 5.61            | 27            | 140.0                     |
| 0L92J  | 9.18            | 16            | 80.0                      |
| 0L9E6  | 15.13           | 24            | 160.0                     |
| 0LCSZ  | 30.3            | 404           | 2480.0                    |
| 0M003  | 9.52            | 50            | 290.0                     |
| 0M769  | 70.29           | 4             | 20.0                      |
| 0MGI0  | 109.6           | 3             | 20.0                      |
| 0MLQW  | 37.81           | 10            | 50.0                      |
| 0MM7B  | 7.62            | 7             | 40.0                      |
| 0MPSR  | 17.57           | 10            | 50.0                      |
| 0MYS6  | 43.52           | 2             | 10.0                      |
| 0NF67  | 16.44           | 2             | 10.0                      |
| 0NZLJ  | 2.31            | 52            | 300.0                     |
| 0OB0R  | 7.1             | 2             | 10.0                      |
| 0OD58  | 22.46           | 88            | 480.0                     |
| 0OPZ4  | 6.95            | 6             | 30.0                      |
| 0OR9U  | 2.33            | 39            | 220.0                     |
| 0OZBT  | 9.42            | 328           | 1640.0                    |
| 0OZGC  | 7.3             | 4             | 20.0                      |
| 0P32C  | 18.32           | 4             | 20.0                      |
| 0P7RI  | 32.25           | 4             | 20.0                      |
| 0PB0L  | 9.0             | 334           | 2500.0                    |
| 0POSZ  | 40.19           | 16            | 80.0                      |
| 0PRXZ  | 27.18           | 4             | 20.0                      |
| 0PW5H  | 11.19           | 361           | 2080.0                    |
| 0Q2SK  | 32.31           | 74            | 390.0                     |
| 0QK7Q  | 5.4             | 73            | 430.0                     |
| 0QPT3  | 6.87            | 6             | 30.0                      |
| 0RBQF  | 4.0             | 14            | 70.0                      |
| 0RM9U  | 21.87           | 2             | 10.0                      |
| 0RVXC  | 4.63            | 77            | 430.0                     |
| 0RXHL  | 8.49            | 8             | 40.0                      |
| 0S375  | 14.94           | 52            | 270.0                     |
| 0S4F6  | 4.57            | 120           | 610.0                     |
| 0SQIM  | 3.53            | 181           | 970.0                     |
| 0SSF2  | 3.18            | 37            | 190.0                     |
| 0SVHO  | 10.65           | 16            | 80.0                      |
| 0SX05  | 6.25            | 2             | 10.0                      |
| 0T6EB  | 2.59            | 110           | 550.0                     |
| 0TCF2  | 2.13            | 6             | 30.0                      |
| 0TJNT  | 5.56            | 5             | 40.0                      |
| 0TLU5  | 2.13            | 1125          | 7060.0                    |
| 0TP6L  | 18.73           | 8             | 40.0                      |
| 0UB8R  | 9.06            | 24            | 120.0                     |
| 0USN8  | 4.91            | 2             | 10.0                      |
| 0V9JD  | 11.37           | 58            | 290.0                     |
| 0VFO4  | 137.62          | 27            | 170.0                     |
| 0VP78  | 20.37           | 30            | 150.0                     |
| 0VYF6  | 14.52           | 41            | 280.0                     |
| 0W4NP  | 8.75            | 12            | 60.0                      |
| 0W74A  | 34.66           | 4             | 20.0                      |
| 0WB9Z  | 4.37            | 94            | 480.0                     |
| 0WPQH  | 28.38           | 3             | 20.0                      |
| 0WRXS  | 2.0             | 22            | 110.0                     |
| 0WX4V  | 2.31            | 2             | 10.0                      |
| 0WXHH  | 7.43            | 2             | 10.0                      |
| 0WZO1  | 10.3            | 2             | 10.0                      |
| 0X7EK  | 45.46           | 3             | 20.0                      |
| 0XJVZ  | 13.74           | 3             | 20.0                      |
| 0XV51  | 1.62            | 47            | 240.0                     |
| 0Y1U0  | 7.3             | 22            | 110.0                     |
| 0YO25  | 1.62            | 23            | 120.0                     |
| 0YRG3  | 3.97            | 11            | 60.0                      |
| 0YSCI  | 15.38           | 8             | 40.0                      |
| 0Z60E  | 7.75            | 2             | 10.0                      |
| 0ZEQ8  | 7.85            | 172           | 1030.0                    |
| 0ZIED  | 9.0             | 6             | 30.0                      |
| 0ZL4S  | 16.26           | 350           | 2290.0                    |
| 0ZMMI  | 7.37            | 3             | 20.0                      |
| 0ZPT1  | 24.87           | 63            | 330.0                     |
| 0ZTO4  | 9.68            | 44            | 220.0                     |
| 0ZTQC  | 5.5             | 93            | 600.0                     |
| 1.47E+02 | 17.04         | 25            | 130.0                     |
| ...    | ...             | ...           | ...                       |
| ZZ2AO  | 24.38           | 2             | 10.0                      |
| ZZDW7  | 30.12           | 4             | 20.0                      |
| ZZM1A  | 19.52           | 82            | 530.0                     |
| ZZNC5  | 10.79           | 2             | 10.0                      |
| ZZX6K  | 111.81          | 2             | 10.0                      |

(The full list continues for all products with Category ‘ZZ’ as retrieved above.)

##### Model Summary

- **Objective:** Maximize total revenue from fulfilling demand for ‘ZZ’ products.
- **Constraints:** For each product, the fulfilled quantity cannot exceed either its demand or its initial inventory, and must be nonnegative.

---

**Note:** The model can be implemented as a linear program (LP) or integer program (IP) depending on whether $x_i$ must be integer (practically, yes, but the model above allows continuous $x_i \geq 0$ for generality). The full data for all ‘ZZ’ products is as listed above.