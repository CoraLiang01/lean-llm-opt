##### Decision Variables

Let $x_i$ be the number of units of pizza type $i$ to be fulfilled, for each pizza type $i$ listed below.  
$x_i \in \mathbb{Z}_{\geq 0}$ (non-negative integer).

##### Parameters

Let $R_i$ be the revenue per unit for pizza type $i$.  
Let $I_i$ be the initial inventory for pizza type $i$.  
Let $D_i$ be the demand for pizza type $i$.

The sets and parameters are as follows (all identifiers and values preserved):

| Pizza Type           | $R_i$ (Revenue) | $I_i$ (Initial Inventory) | $D_i$ (Demand) |
|----------------------|-----------------|--------------------------|----------------|
| bbq_ckn_l            | 20.75           | 9920                     | 1960           |
| bbq_ckn_m            | 16.75           | 9560                     | 1884           |
| bbq_ckn_s            | 12.75           | 4840                     | 963            |
| big_meat_s           | 12              | 19140                    | 3729           |
| brie_carre_s         | 23.65           | 4900                     | 970            |
| calabrese_l          | 20.25           | 2760                     | 550            |
| calabrese_m          | 16.25           | 5620                     | 1116           |
| calabrese_s          | 12.25           | 990                      | 198            |
| cali_ckn_l           | 20.75           | 9270                     | 1823           |
| cali_ckn_m           | 16.75           | 9440                     | 1858           |
| cali_ckn_s           | 12.75           | 4990                     | 992            |
| ckn_alfredo_l        | 20.75           | 1880                     | 375            |
| ckn_alfredo_m        | 16.75           | 7030                     | 1400           |
| ckn_alfredo_s        | 12.75           | 960                      | 192            |
| ckn_pesto_l          | 20.75           | 3990                     | 791            |
| ckn_pesto_m          | 16.75           | 2760                     | 550            |
| ckn_pesto_s          | 12.75           | 2980                     | 593            |
| classic_dlx_l        | 20.5            | 4730                     | 944            |
| classic_dlx_m        | 16              | 11810                    | 2340           |
| classic_dlx_s        | 12              | 7990                     | 1586           |
| five_cheese_l        | 18.5            | 14090                    | 2769           |
| four_cheese_l        | 17.95           | 13160                    | 2589           |
| four_cheese_m        | 14.75           | 5860                     | 1163           |
| green_garden_l       | 20.25           | 950                      | 189            |
| green_garden_m       | 16              | 3020                     | 602            |
| green_garden_s       | 12              | 6000                     | 1193           |
| hawaiian_l           | 16.5            | 9190                     | 1815           |
| hawaiian_m           | 13.25           | 4830                     | 956            |
| hawaiian_s           | 10.5            | 10200                    | 2021           |
| ital_cpcllo_l        | 20.5            | 7320                     | 1447           |
| ital_cpcllo_m        | 16              | 4040                     | 803            |
| ital_cpcllo_s        | 12              | 3020                     | 602            |
| ital_supr_l          | 20.75           | 7470                     | 1482           |
| ital_supr_m          | 16.5            | 9410                     | 1861           |
| ital_supr_s          | 12.5            | 1960                     | 390            |
| ital_veggie_l        | 21              | 1900                     | 380            |
| ital_veggie_m        | 16.75           | 4860                     | 969            |
| ital_veggie_s        | 12.75           | 3050                     | 607            |
| mediterraneo_l       | 20.25           | 3700                     | 734            |
| mediterraneo_m       | 16              | 2750                     | 546            |
| mediterraneo_s       | 12              | 2890                     | 577            |
| mexicana_l           | 20.25           | 8670                     | 1711           |
| mexicana_m           | 16              | 4550                     | 907            |
| mexicana_s           | 12              | 1620                     | 322            |
| napolitana_l         | 20.5            | 5660                     | 1123           |
| napolitana_m         | 16              | 4270                     | 853            |
| napolitana_s         | 12              | 4710                     | 939            |
| pep_msh_pep_l        | 17.5            | 3840                     | 765            |
| pep_msh_pep_m        | 14.5            | 3970                     | 788            |
| pep_msh_pep_s        | 11              | 5780                     | 1148           |
| pepperoni_l          | 15.25           | 7280                     | 1440           |
| pepperoni_m          | 12.5            | 9390                     | 1857           |
| pepperoni_s          | 9.75            | 7510                     | 1490           |
| peppr_salami_l       | 20.75           | 6960                     | 1376           |
| peppr_salami_m       | 16.5            | 4280                     | 852            |
| peppr_salami_s       | 12.5            | 3220                     | 640            |
| prsc_argla_l         | 20.75           | 4350                     | 858            |
| prsc_argla_m         | 16.5            | 5980                     | 1183           |
| prsc_argla_s         | 12.5            | 4240                     | 844            |
| sicilian_l           | 20.25           | 6130                     | 1209           |
| sicilian_m           | 16.25           | 5740                     | 1135           |
| sicilian_s           | 12.25           | 7510                     | 1482           |
| soppressata_l        | 20.75           | 4050                     | 806            |
| soppressata_m        | 16.5            | 2680                     | 536            |
| soppressata_s        | 12.5            | 2880                     | 576            |
| southw_ckn_l         | 20.75           | 10160                    | 2009           |
| southw_ckn_m         | 16.75           | 5340                     | 1060           |
| southw_ckn_s         | 12.75           | 3670                     | 733            |
| spicy_ital_l         | 20.75           | 11090                    | 2198           |
| spicy_ital_m         | 16.5            | 4080                     | 808            |
| spicy_ital_s         | 12.5            | 4070                     | 807            |
| spin_pesto_l         | 20.75           | 2840                     | 563            |
| spin_pesto_m         | 16.5            | 2820                     | 563            |
| spin_pesto_s         | 12.5            | 4040                     | 801            |
| spinach_fet_l        | 20.25           | 4450                     | 882            |
| spinach_fet_m        | 16              | 5620                     | 1120           |
| spinach_fet_s        | 12              | 4390                     | 876            |
| spinach_supr_l       | 20.75           | 2830                     | 563            |
| spinach_supr_m       | 16.5            | 2670                     | 533            |
| spinach_supr_s       | 12.5            | 4000                     | 794            |
| thai_ckn_l           | 20.75           | 14100                    | 2776           |
| thai_ckn_m           | 16.75           | 4810                     | 955            |
| thai_ckn_s           | 12.75           | 4800                     | 956            |
| the_greek_l          | 20.5            | 2550                     | 510            |
| the_greek_m          | 16              | 2810                     | 560            |
| the_greek_s          | 12              | 3040                     | 604            |
| the_greek_xl         | 25.5            | 5520                     | 1096           |
| the_greek_xxl        | 35.95           | 280                      | 56             |
| veggie_veg_l         | 20.25           | 4270                     | 850            |
| veggie_veg_m         | 16              | 6350                     | 1265           |
| veggie_veg_s         | 12              | 4640                     | 921            |
| fried_duck_l         | 16.5            | 2830                     | 733            |
| fried_duck_m         | 12.5            | 2670                     | 2198           |
| fried_duck_s         | 21              | 4000                     | 808            |
| mapo_Tofu_l          | 16.75           | 2680                     | 563            |
| mapo_Tofu_m          | 12.75           | 2880                     | 801            |
| mapo_Tofu_s          | 20.25           | 10160                    | 882            |
| sea_food_l           | 16              | 2820                     | 1120           |
| sea_food_m           | 20.75           | 4040                     | 794            |
| sea_food_s           | 16.5            | 4450                     | 2776           |

##### Objective Function

\[
\max \sum_{i} R_i x_i
\]

##### Constraints

For each pizza type $i$:
- Inventory and demand fulfillment:
  \[
  0 \leq x_i \leq \min\{I_i, D_i\}
  \]
  (i.e., $x_i$ cannot exceed either the available initial inventory or the demand for that pizza type.)

- Integer domain:
  \[
  x_i \in \mathbb{Z}_{\geq 0}
  \]

##### Complete Mathematical Model

\[
\begin{align*}
\max \quad & \sum_{i} R_i x_i \\
\text{s.t.} \quad & 0 \leq x_i \leq \min\{I_i, D_i\}, \quad \forall i \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i
\end{align*}
\]

Where all $i$ and their corresponding $R_i$, $I_i$, $D_i$ are as listed above. All identifiers and values are preserved as in the dataset.