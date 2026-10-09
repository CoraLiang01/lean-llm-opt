**Index Set:**  
Let $\mathcal{I}$ be the set of all pizza types, indexed by $i$.

$\mathcal{I} = \{$  
bbq_ckn_l,  
bbq_ckn_m,  
bbq_ckn_s,  
big_meat_s,  
brie_carre_s,  
calabrese_l,  
calabrese_m,  
calabrese_s,  
cali_ckn_l,  
cali_ckn_m,  
cali_ckn_s,  
ckn_alfredo_l,  
ckn_alfredo_m,  
ckn_alfredo_s,  
ckn_pesto_l,  
ckn_pesto_m,  
ckn_pesto_s,  
classic_dlx_l,  
classic_dlx_m,  
classic_dlx_s,  
five_cheese_l,  
four_cheese_l,  
four_cheese_m,  
green_garden_l,  
green_garden_m,  
green_garden_s,  
hawaiian_l,  
hawaiian_m,  
hawaiian_s,  
ital_cpcllo_l,  
ital_cpcllo_m,  
ital_cpcllo_s,  
ital_supr_l,  
ital_supr_m,  
ital_supr_s,  
ital_veggie_l,  
ital_veggie_m,  
ital_veggie_s,  
mediterraneo_l,  
mediterraneo_m,  
mediterraneo_s,  
mexicana_l,  
mexicana_m,  
mexicana_s,  
napolitana_l,  
napolitana_m,  
napolitana_s,  
pep_msh_pep_l,  
pep_msh_pep_m,  
pep_msh_pep_s,  
pepperoni_l,  
pepperoni_m,  
pepperoni_s,  
peppr_salami_l,  
peppr_salami_m,  
peppr_salami_s,  
prsc_argla_l,  
prsc_argla_m,  
prsc_argla_s,  
sicilian_l,  
sicilian_m,  
sicilian_s,  
soppressata_l,  
soppressata_m,  
soppressata_s,  
southw_ckn_l,  
southw_ckn_m,  
southw_ckn_s,  
spicy_ital_l,  
spicy_ital_m,  
spicy_ital_s,  
spin_pesto_l,  
spin_pesto_m,  
spin_pesto_s,  
spinach_fet_l,  
spinach_fet_m,  
spinach_fet_s,  
spinach_supr_l,  
spinach_supr_m,  
spinach_supr_s,  
thai_ckn_l,  
thai_ckn_m,  
thai_ckn_s,  
the_greek_l,  
the_greek_m,  
the_greek_s,  
the_greek_xl,  
the_greek_xxl,  
veggie_veg_l,  
veggie_veg_m,  
veggie_veg_s,  
fried_duck_l,  
fried_duck_m,  
fried_duck_s,  
mapo_Tofu_l,  
mapo_Tofu_m,  
mapo_Tofu_s,  
sea_food_l,  
sea_food_m,  
sea_food_s  
$\}$

**Parameters:**  
For each $i \in \mathcal{I}$:

- $A_i$: Revenue per unit of pizza type $i$  
- $d_i$: Demand for pizza type $i$  
- $I_i$: Initial inventory for pizza type $i$  

Parameter values (in source order):

| $i$                  | $A_i$   | $d_i$ | $I_i$  |
|----------------------|---------|-------|--------|
| bbq_ckn_l            | 20.75   | 1960  | 9920   |
| bbq_ckn_m            | 16.75   | 1884  | 9560   |
| bbq_ckn_s            | 12.75   | 963   | 4840   |
| big_meat_s           | 12      | 3729  | 19140  |
| brie_carre_s         | 23.65   | 970   | 4900   |
| calabrese_l          | 20.25   | 550   | 2760   |
| calabrese_m          | 16.25   | 1116  | 5620   |
| calabrese_s          | 12.25   | 198   | 990    |
| cali_ckn_l           | 20.75   | 1823  | 9270   |
| cali_ckn_m           | 16.75   | 1858  | 9440   |
| cali_ckn_s           | 12.75   | 992   | 4990   |
| ckn_alfredo_l        | 20.75   | 375   | 1880   |
| ckn_alfredo_m        | 16.75   | 1400  | 7030   |
| ckn_alfredo_s        | 12.75   | 192   | 960    |
| ckn_pesto_l          | 20.75   | 791   | 3990   |
| ckn_pesto_m          | 16.75   | 550   | 2760   |
| ckn_pesto_s          | 12.75   | 593   | 2980   |
| classic_dlx_l        | 20.5    | 944   | 4730   |
| classic_dlx_m        | 16      | 2340  | 11810  |
| classic_dlx_s        | 12      | 1586  | 7990   |
| five_cheese_l        | 18.5    | 2769  | 14090  |
| four_cheese_l        | 17.95   | 2589  | 13160  |
| four_cheese_m        | 14.75   | 1163  | 5860   |
| green_garden_l       | 20.25   | 189   | 950    |
| green_garden_m       | 16      | 602   | 3020   |
| green_garden_s       | 12      | 1193  | 6000   |
| hawaiian_l           | 16.5    | 1815  | 9190   |
| hawaiian_m           | 13.25   | 956   | 4830   |
| hawaiian_s           | 10.5    | 2021  | 10200  |
| ital_cpcllo_l        | 20.5    | 1447  | 7320   |
| ital_cpcllo_m        | 16      | 803   | 4040   |
| ital_cpcllo_s        | 12      | 602   | 3020   |
| ital_supr_l          | 20.75   | 1482  | 7470   |
| ital_supr_m          | 16.5    | 1861  | 9410   |
| ital_supr_s          | 12.5    | 390   | 1960   |
| ital_veggie_l        | 21      | 380   | 1900   |
| ital_veggie_m        | 16.75   | 969   | 4860   |
| ital_veggie_s        | 12.75   | 607   | 3050   |
| mediterraneo_l       | 20.25   | 734   | 3700   |
| mediterraneo_m       | 16      | 546   | 2750   |
| mediterraneo_s       | 12      | 577   | 2890   |
| mexicana_l           | 20.25   | 1711  | 8670   |
| mexicana_m           | 16      | 907   | 4550   |
| mexicana_s           | 12      | 322   | 1620   |
| napolitana_l         | 20.5    | 1123  | 5660   |
| napolitana_m         | 16      | 853   | 4270   |
| napolitana_s         | 12      | 939   | 4710   |
| pep_msh_pep_l        | 17.5    | 765   | 3840   |
| pep_msh_pep_m        | 14.5    | 788   | 3970   |
| pep_msh_pep_s        | 11      | 1148  | 5780   |
| pepperoni_l          | 15.25   | 1440  | 7280   |
| pepperoni_m          | 12.5    | 1857  | 9390   |
| pepperoni_s          | 9.75    | 1490  | 7510   |
| peppr_salami_l       | 20.75   | 1376  | 6960   |
| peppr_salami_m       | 16.5    | 852   | 4280   |
| peppr_salami_s       | 12.5    | 640   | 3220   |
| prsc_argla_l         | 20.75   | 858   | 4350   |
| prsc_argla_m         | 16.5    | 1183  | 5980   |
| prsc_argla_s         | 12.5    | 844   | 4240   |
| sicilian_l           | 20.25   | 1209  | 6130   |
| sicilian_m           | 16.25   | 1135  | 5740   |
| sicilian_s           | 12.25   | 1482  | 7510   |
| soppressata_l        | 20.75   | 806   | 4050   |
| soppressata_m        | 16.5    | 536   | 2680   |
| soppressata_s        | 12.5    | 576   | 2880   |
| southw_ckn_l         | 20.75   | 2009  | 10160  |
| southw_ckn_m         | 16.75   | 1060  | 5340   |
| southw_ckn_s         | 12.75   | 733   | 3670   |
| spicy_ital_l         | 20.75   | 2198  | 11090  |
| spicy_ital_m         | 16.5    | 808   | 4080   |
| spicy_ital_s         | 12.5    | 807   | 4070   |
| spin_pesto_l         | 20.75   | 563   | 2840   |
| spin_pesto_m         | 16.5    | 563   | 2820   |
| spin_pesto_s         | 12.5    | 801   | 4040   |
| spinach_fet_l        | 20.25   | 882   | 4450   |
| spinach_fet_m        | 16      | 1120  | 5620   |
| spinach_fet_s        | 12      | 876   | 4390   |
| spinach_supr_l       | 20.75   | 563   | 2830   |
| spinach_supr_m       | 16.5    | 533   | 2670   |
| spinach_supr_s       | 12.5    | 794   | 4000   |
| thai_ckn_l           | 20.75   | 2776  | 14100  |
| thai_ckn_m           | 16.75   | 955   | 4810   |
| thai_ckn_s           | 12.75   | 956   | 4800   |
| the_greek_l          | 20.5    | 510   | 2550   |
| the_greek_m          | 16      | 560   | 2810   |
| the_greek_s          | 12      | 604   | 3040   |
| the_greek_xl         | 25.5    | 1096  | 5520   |
| the_greek_xxl        | 35.95   | 56    | 280    |
| veggie_veg_l         | 20.25   | 850   | 4270   |
| veggie_veg_m         | 16      | 1265  | 6350   |
| veggie_veg_s         | 12      | 921   | 4640   |
| fried_duck_l         | 16.5    | 733   | 2830   |
| fried_duck_m         | 12.5    | 2198  | 2670   |
| fried_duck_s         | 21      | 808   | 4000   |
| mapo_Tofu_l          | 16.75   | 563   | 2680   |
| mapo_Tofu_m          | 12.75   | 801   | 2880   |
| mapo_Tofu_s          | 20.25   | 882   | 10160  |
| sea_food_l           | 16      | 1120  | 2820   |
| sea_food_m           | 20.75   | 794   | 4040   |
| sea_food_s           | 16.5    | 2776  | 4450   |

**Decision Variables:**  
For each $i \in \mathcal{I}$:  
$x_i$ = number of units of pizza type $i$ to fulfill (integer, $x_i \geq 0$)

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
$$

**Constraints:**  
For each $i \in \mathcal{I}$:
1. Inventory constraint:  
   $x_i \leq I_i$
2. Demand constraint:  
   $x_i \leq d_i$
3. Non-negativity and integrality:  
   $x_i \in \mathbb{Z}_{\geq 0}$

**Full Model:**

$$
\begin{align*}
\max_{x_i \in \mathbb{Z}_{\geq 0},\, i \in \mathcal{I}} \quad & \sum_{i \in \mathcal{I}} A_i \cdot x_i \\
\text{s.t.} \quad & x_i \leq I_i, \quad \forall i \in \mathcal{I} \\
                  & x_i \leq d_i, \quad \forall i \in \mathcal{I} \\
                  & x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in \mathcal{I}
\end{align*}
$$

**Parameter Table (source order):**

| $i$                  | $A_i$   | $d_i$ | $I_i$  |
|----------------------|---------|-------|--------|
| bbq_ckn_l            | 20.75   | 1960  | 9920   |
| bbq_ckn_m            | 16.75   | 1884  | 9560   |
| bbq_ckn_s            | 12.75   | 963   | 4840   |
| big_meat_s           | 12      | 3729  | 19140  |
| brie_carre_s         | 23.65   | 970   | 4900   |
| calabrese_l          | 20.25   | 550   | 2760   |
| calabrese_m          | 16.25   | 1116  | 5620   |
| calabrese_s          | 12.25   | 198   | 990    |
| cali_ckn_l           | 20.75   | 1823  | 9270   |
| cali_ckn_m           | 16.75   | 1858  | 9440   |
| cali_ckn_s           | 12.75   | 992   | 4990   |
| ckn_alfredo_l        | 20.75   | 375   | 1880   |
| ckn_alfredo_m        | 16.75   | 1400  | 7030   |
| ckn_alfredo_s        | 12.75   | 192   | 960    |
| ckn_pesto_l          | 20.75   | 791   | 3990   |
| ckn_pesto_m          | 16.75   | 550   | 2760   |
| ckn_pesto_s          | 12.75   | 593   | 2980   |
| classic_dlx_l        | 20.5    | 944   | 4730   |
| classic_dlx_m        | 16      | 2340  | 11810  |
| classic_dlx_s        | 12      | 1586  | 7990   |
| five_cheese_l        | 18.5    | 2769  | 14090  |
| four_cheese_l        | 17.95   | 2589  | 13160  |
| four_cheese_m        | 14.75   | 1163  | 5860   |
| green_garden_l       | 20.25   | 189   | 950    |
| green_garden_m       | 16      | 602   | 3020   |
| green_garden_s       | 12      | 1193  | 6000   |
| hawaiian_l           | 16.5    | 1815  | 9190   |
| hawaiian_m           | 13.25   | 956   | 4830   |
| hawaiian_s           | 10.5    | 2021  | 10200  |
| ital_cpcllo_l        | 20.5    | 1447  | 7320   |
| ital_cpcllo_m        | 16      | 803   | 4040   |
| ital_cpcllo_s        | 12      | 602   | 3020   |
| ital_supr_l          | 20.75   | 1482  | 7470   |
| ital_supr_m          | 16.5    | 1861  | 9410   |
| ital_supr_s          | 12.5    | 390   | 1960   |
| ital_veggie_l        | 21      | 380   | 1900   |
| ital_veggie_m        | 16.75   | 969   | 4860   |
| ital_veggie_s        | 12.75   | 607   | 3050   |
| mediterraneo_l       | 20.25   | 734   | 3700   |
| mediterraneo_m       | 16      | 546   | 2750   |
| mediterraneo_s       | 12      | 577   | 2890   |
| mexicana_l           | 20.25   | 1711  | 8670   |
| mexicana_m           | 16      | 907   | 4550   |
| mexicana_s           | 12      | 322   | 1620   |
| napolitana_l         | 20.5    | 1123  | 5660   |
| napolitana_m         | 16      | 853   | 4270   |
| napolitana_s         | 12      | 939   | 4710   |
| pep_msh_pep_l        | 17.5    | 765   | 3840   |
| pep_msh_pep_m        | 14.5    | 788   | 3970   |
| pep_msh_pep_s        | 11      | 1148  | 5780   |
| pepperoni_l          | 15.25   | 1440  | 7280   |
| pepperoni_m          | 12.5    | 1857  | 9390   |
| pepperoni_s          | 9.75    | 1490  | 7510   |
| peppr_salami_l       | 20.75   | 1376  | 6960   |
| peppr_salami_m       | 16.5    | 852   | 4280   |
| peppr_salami_s       | 12.5    | 640   | 3220   |
| prsc_argla_l         | 20.75   | 858   | 4350   |
| prsc_argla_m         | 16.5    | 1183  | 5980   |
| prsc_argla_s         | 12.5    | 844   | 4240   |
| sicilian_l           | 20.25   | 1209  | 6130   |
| sicilian_m           | 16.25   | 1135  | 5740   |
| sicilian_s           | 12.25   | 1482  | 7510   |
| soppressata_l        | 20.75   | 806   | 4050   |
| soppressata_m        | 16.5    | 536   | 2680   |
| soppressata_s        | 12.5    | 576   | 2880   |
| southw_ckn_l         | 20.75   | 2009  | 10160  |
| southw_ckn_m         | 16.75   | 1060  | 5340   |
| southw_ckn_s         | 12.75   | 733   | 3670   |
| spicy_ital_l         | 20.75   | 2198  | 11090  |
| spicy_ital_m         | 16.5    | 808   | 4080   |
| spicy_ital_s         | 12.5    | 807   | 4070   |
| spin_pesto_l         | 20.75   | 563   | 2840   |
| spin_pesto_m         | 16.5    | 563   | 2820   |
| spin_pesto_s         | 12.5    | 801   | 4040   |
| spinach_fet_l        | 20.25   | 882   | 4450   |
| spinach_fet_m        | 16      | 1120  | 5620   |
| spinach_fet_s        | 12      | 876   | 4390   |
| spinach_supr_l       | 20.75   | 563   | 2830   |
| spinach_supr_m       | 16.5    | 533   | 2670   |
| spinach_supr_s       | 12.5    | 794   | 4000   |
| thai_ckn_l           | 20.75   | 2776  | 14100  |
| thai_ckn_m           | 16.75   | 955   | 4810   |
| thai_ckn_s           | 12.75   | 956   | 4800   |
| the_greek_l          | 20.5    | 510   | 2550   |
| the_greek_m          | 16      | 560   | 2810   |
| the_greek_s          | 12      | 604   | 3040   |
| the_greek_xl         | 25.5    | 1096  | 5520   |
| the_greek_xxl        | 35.95   | 56    | 280    |
| veggie_veg_l         | 20.25   | 850   | 4270   |
| veggie_veg_m         | 16      | 1265  | 6350   |
| veggie_veg_s         | 12      | 921   | 4640   |
| fried_duck_l         | 16.5    | 733   | 2830   |
| fried_duck_m         | 12.5    | 2198  | 2670   |
| fried_duck_s         | 21      | 808   | 4000   |
| mapo_Tofu_l          | 16.75   | 563   | 2680   |
| mapo_Tofu_m          | 12.75   | 801   | 2880   |
| mapo_Tofu_s          | 20.25   | 882   | 10160  |
| sea_food_l           | 16      | 1120  | 2820   |
| sea_food_m           | 20.75   | 794   | 4040   |
| sea_food_s           | 16.5    | 2776  | 4450   |