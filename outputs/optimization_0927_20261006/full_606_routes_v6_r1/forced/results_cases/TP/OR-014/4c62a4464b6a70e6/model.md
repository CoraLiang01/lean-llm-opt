Let $I$ be the set of all pizza types, indexed by $i$, with the following data for each $i$:

- Product Name: $p_i$
- Revenue per unit: $r_i$
- Demand: $d_i$
- Initial Inventory: $s_i$

Define decision variables:

$x_i \in \mathbb{Z}_+, \quad \forall i \in I$ (number of units of pizza type $i$ to fulfill)

Objective:
$$
\max \sum_{i \in I} r_i x_i
$$

Subject to:
$$
0 \leq x_i \leq \min\{d_i, s_i\}, \quad \forall i \in I
$$
$$
x_i \in \mathbb{Z}_+, \quad \forall i \in I
$$

Where the data for each $i$ is as follows (in source order):

| $i$ | $p_i$              | $r_i$  | $d_i$ | $s_i$  |
|-----|--------------------|--------|-------|--------|
| 1   | bbq_ckn_l          | 20.75  | 1960  | 9920   |
| 2   | bbq_ckn_m          | 16.75  | 1884  | 9560   |
| 3   | bbq_ckn_s          | 12.75  | 963   | 4840   |
| 4   | big_meat_s         | 12     | 3729  | 19140  |
| 5   | brie_carre_s       | 23.65  | 970   | 4900   |
| 6   | calabrese_l        | 20.25  | 550   | 2760   |
| 7   | calabrese_m        | 16.25  | 1116  | 5620   |
| 8   | calabrese_s        | 12.25  | 198   | 990    |
| 9   | cali_ckn_l         | 20.75  | 1823  | 9270   |
| 10  | cali_ckn_m         | 16.75  | 1858  | 9440   |
| 11  | cali_ckn_s         | 12.75  | 992   | 4990   |
| 12  | ckn_alfredo_l      | 20.75  | 375   | 1880   |
| 13  | ckn_alfredo_m      | 16.75  | 1400  | 7030   |
| 14  | ckn_alfredo_s      | 12.75  | 192   | 960    |
| 15  | ckn_pesto_l        | 20.75  | 791   | 3990   |
| 16  | ckn_pesto_m        | 16.75  | 550   | 2760   |
| 17  | ckn_pesto_s        | 12.75  | 593   | 2980   |
| 18  | classic_dlx_l      | 20.5   | 944   | 4730   |
| 19  | classic_dlx_m      | 16     | 2340  | 11810  |
| 20  | classic_dlx_s      | 12     | 1586  | 7990   |
| 21  | five_cheese_l      | 18.5   | 2769  | 14090  |
| 22  | four_cheese_l      | 17.95  | 2589  | 13160  |
| 23  | four_cheese_m      | 14.75  | 1163  | 5860   |
| 24  | green_garden_l     | 20.25  | 189   | 950    |
| 25  | green_garden_m     | 16     | 602   | 3020   |
| 26  | green_garden_s     | 12     | 1193  | 6000   |
| 27  | hawaiian_l         | 16.5   | 1815  | 9190   |
| 28  | hawaiian_m         | 13.25  | 956   | 4830   |
| 29  | hawaiian_s         | 10.5   | 2021  | 10200  |
| 30  | ital_cpcllo_l      | 20.5   | 1447  | 7320   |
| 31  | ital_cpcllo_m      | 16     | 803   | 4040   |
| 32  | ital_cpcllo_s      | 12     | 602   | 3020   |
| 33  | ital_supr_l        | 20.75  | 1482  | 7470   |
| 34  | ital_supr_m        | 16.5   | 1861  | 9410   |
| 35  | ital_supr_s        | 12.5   | 390   | 1960   |
| 36  | ital_veggie_l      | 21     | 380   | 1900   |
| 37  | ital_veggie_m      | 16.75  | 969   | 4860   |
| 38  | ital_veggie_s      | 12.75  | 607   | 3050   |
| 39  | mediterraneo_l     | 20.25  | 734   | 3700   |
| 40  | mediterraneo_m     | 16     | 546   | 2750   |
| 41  | mediterraneo_s     | 12     | 577   | 2890   |
| 42  | mexicana_l         | 20.25  | 1711  | 8670   |
| 43  | mexicana_m         | 16     | 907   | 4550   |
| 44  | mexicana_s         | 12     | 322   | 1620   |
| 45  | napolitana_l       | 20.5   | 1123  | 5660   |
| 46  | napolitana_m       | 16     | 853   | 4270   |
| 47  | napolitana_s       | 12     | 939   | 4710   |
| 48  | pep_msh_pep_l      | 17.5   | 765   | 3840   |
| 49  | pep_msh_pep_m      | 14.5   | 788   | 3970   |
| 50  | pep_msh_pep_s      | 11     | 1148  | 5780   |
| 51  | pepperoni_l        | 15.25  | 1440  | 7280   |
| 52  | pepperoni_m        | 12.5   | 1857  | 9390   |
| 53  | pepperoni_s        | 9.75   | 1490  | 7510   |
| 54  | peppr_salami_l     | 20.75  | 1376  | 6960   |
| 55  | peppr_salami_m     | 16.5   | 852   | 4280   |
| 56  | peppr_salami_s     | 12.5   | 640   | 3220   |
| 57  | prsc_argla_l       | 20.75  | 858   | 4350   |
| 58  | prsc_argla_m       | 16.5   | 1183  | 5980   |
| 59  | prsc_argla_s       | 12.5   | 844   | 4240   |
| 60  | sicilian_l         | 20.25  | 1209  | 6130   |
| 61  | sicilian_m         | 16.25  | 1135  | 5740   |
| 62  | sicilian_s         | 12.25  | 1482  | 7510   |
| 63  | soppressata_l      | 20.75  | 806   | 4050   |
| 64  | soppressata_m      | 16.5   | 536   | 2680   |
| 65  | soppressata_s      | 12.5   | 576   | 2880   |
| 66  | southw_ckn_l       | 20.75  | 2009  | 10160  |
| 67  | southw_ckn_m       | 16.75  | 1060  | 5340   |
| 68  | southw_ckn_s       | 12.75  | 733   | 3670   |
| 69  | spicy_ital_l       | 20.75  | 2198  | 11090  |
| 70  | spicy_ital_m       | 16.5   | 808   | 4080   |
| 71  | spicy_ital_s       | 12.5   | 807   | 4070   |
| 72  | spin_pesto_l       | 20.75  | 563   | 2840   |
| 73  | spin_pesto_m       | 16.5   | 563   | 2820   |
| 74  | spin_pesto_s       | 12.5   | 801   | 4040   |
| 75  | spinach_fet_l      | 20.25  | 882   | 4450   |
| 76  | spinach_fet_m      | 16     | 1120  | 5620   |
| 77  | spinach_fet_s      | 12     | 876   | 4390   |
| 78  | spinach_supr_l     | 20.75  | 563   | 2830   |
| 79  | spinach_supr_m     | 16.5   | 533   | 2670   |
| 80  | spinach_supr_s     | 12.5   | 794   | 4000   |
| 81  | thai_ckn_l         | 20.75  | 2776  | 14100  |
| 82  | thai_ckn_m         | 16.75  | 955   | 4810   |
| 83  | thai_ckn_s         | 12.75  | 956   | 4800   |
| 84  | the_greek_l        | 20.5   | 510   | 2550   |
| 85  | the_greek_m        | 16     | 560   | 2810   |
| 86  | the_greek_s        | 12     | 604   | 3040   |
| 87  | the_greek_xl       | 25.5   | 1096  | 5520   |
| 88  | the_greek_xxl      | 35.95  | 56    | 280    |
| 89  | veggie_veg_l       | 20.25  | 850   | 4270   |
| 90  | veggie_veg_m       | 16     | 1265  | 6350   |
| 91  | veggie_veg_s       | 12     | 921   | 4640   |
| 92  | fried_duck_l       | 16.5   | 733   | 2830   |
| 93  | fried_duck_m       | 12.5   | 2198  | 2670   |
| 94  | fried_duck_s       | 21     | 808   | 4000   |
| 95  | mapo_Tofu_l        | 16.75  | 563   | 2680   |
| 96  | mapo_Tofu_m        | 12.75  | 801   | 2880   |
| 97  | mapo_Tofu_s        | 20.25  | 882   | 10160  |
| 98  | sea_food_l         | 16     | 1120  | 2820   |
| 99  | sea_food_m         | 20.75  | 794   | 4040   |
| 100 | sea_food_s         | 16.5   | 2776  | 4450   |

All data is preserved in source order and with original identifiers and coefficients.