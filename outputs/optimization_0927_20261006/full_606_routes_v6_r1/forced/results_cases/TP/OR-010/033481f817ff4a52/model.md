Let $I$ be the set of products, indexed in the order given below. For each product $i\in I$, let:

- $x_i$: integer, number of units of product $i$ fulfilled ($x_i\geq0$ and integer).

Parameters for each product $i$ (from the data):

- $r_i$: revenue per unit (from "Revenue" column)
- $d_i$: demand (from "Demand" column)
- $s_i$: initial inventory (from "Initial Inventory" column)

The model is:

$\max \sum_{i\in I} r_i x_i$

subject to

$0 \leq x_i \leq \min\{d_i,\, s_i\}$, integer, for all $i\in I$

where all data is as follows (in source order):

| $i$ | Product Name | $r_i$ (Revenue) | $d_i$ (Demand) | $s_i$ (Initial Inventory) |
|---|-------------------------------|----------------|----------------|------------------------|
| 1 | Adams Group_service | 1003.56 | 10 | 80 |
| 2 | Anderson-Leach_against | 549.92 | 6 | 40 |
| 3 | Anderson-Valdez_somebody | 169.7 | 75 | 560 |
| 4 | Anderson-White_son | 1285.81 | 121 | 820 |
| 5 | Andrews LLC_matter | 1269.71 | 130 | 960 |
| 6 | Andrews PLC_enter | 805.82 | 112 | 850 |
| 7 | Andrews-Martin_build | 1134.19 | 125 | 950 |
| 8 | Arias-Mendoza_life | 1479.23 | 54 | 360 |
| 9 | Bennett and Sons_down | 139.14 | 75 | 580 |
| 10 | Bennett, Foster and Moreno_enter | 518.67 | 102 | 800 |
| 11 | Fernandez, Long and Nelson_member | 1088.57 | 48 | 340 |
| 12 | Fernandez-Fischer_million | 371.56 | 4 | 30 |
| 13 | Ferrell Inc_prove | 956.78 | 125 | 990 |
| 14 | Fields, Christensen and Daniels_nor | 1283.7 | 9 | 70 |
| 15 | Figueroa LLC_involve | 1184.54 | 132 | 890 |
| 16 | Finley Group_itself | 377.8 | 29 | 240 |
| 17 | Fisher Ltd_speak | 701.97 | 43 | 290 |
| 18 | Fisher-Marshall_cover | 875.91 | 5 | 40 |
| 19 | Fuller-Torres_behavior | 951.85 | 40 | 300 |
| 20 | Fuller-Walters_radio | 869.95 | 48 | 350 |
| 21 | Gallagher-Campbell_dark | 887.06 | 87 | 600 |
| 22 | Gallagher-Kirby_talk | 1318.33 | 79 | 610 |
| 23 | Gallagher-Parker_recognize | 448.61 | 20 | 160 |
| 24 | Gonzalez PLC_task | 167.99 | 33 | 260 |
| 25 | Gonzalez, Coleman and Le_heavy | 1068.1 | 60 | 480 |
| 26 | Gonzalez, Lowe and Robinson_education | 872.16 | 25 | 200 |
| 27 | Gonzalez-Horn_light | 483.26 | 42 | 290 |
| 28 | Good, Davis and Smith_station | 1279.24 | 52 | 350 |
| 29 | Goodman, Hughes and White_realize | 894.34 | 116 | 900 |
| 30 | Goodwin PLC_bad | 367.91 | 95 | 760 |
| 31 | Gordon PLC_detail | 178.22 | 66 | 440 |
| 32 | Graham LLC_stage | 1030.8 | 25 | 170 |
| 33 | Graham Ltd_marriage | 479.44 | 78 | 600 |
| 34 | Graham-Swanson_message | 973.62 | 138 | 920 |
| 35 | Grant, Cross and Bennett_religious | 1211.35 | 32 | 240 |
| 36 | Grant, Mcdonald and Watson_owner | 467.93 | 28 | 190 |
| 37 | Graves, Turner and Crawford_wait | 678.72 | 84 | 600 |
| 38 | Gray, Smith and Barnes_much | 478.36 | 102 | 760 |
| 39 | Green Inc_direction | 1196.95 | 112 | 850 |
| 40 | Green-Rogers_could | 1118.63 | 7 | 50 |
| 41 | Greene-Baxter_them | 972.1 | 62 | 470 |
| 42 | Greer and Sons_keep | 338.6 | 109 | 790 |
| 43 | Griffin, Boyle and Dawson_anything | 178.52 | 81 | 580 |
| 44 | Harper Inc_well | 992.14 | 131 | 880 |
| 45 | Harper Ltd_off | 498.77 | 26 | 170 |
| 46 | Harrington, Sosa and Mccarty_coach | 965.34 | 109 | 790 |
| 47 | Harris Group_different | 321.11 | 65 | 500 |
| 48 | Harris and Sons_audience | 287.36 | 121 | 890 |
| 49 | Harris and Sons_fear | 462.26 | 90 | 680 |
| 50 | Harris, Hamilton and Rose_contain | 641.22 | 31 | 250 |
| 51 | Harris, Stevens and Hall_answer | 1136.87 | 24 | 170 |
| 52 | Harris-Bell_painting | 1385.88 | 122 | 830 |
| 53 | Harris-Melton_often | 867.3 | 101 | 680 |
| 54 | Harris-Rogers_after | 1175.92 | 76 | 590 |
| 55 | Hart Group_cold | 153.61 | 100 | 700 |
| 56 | Hutchinson, Roberts and Mcbride_among | 753.55 | 125 | 830 |
| 57 | Ibarra, Jackson and Potter_upon | 245.81 | 45 | 360 |
| 58 | Jackson Inc_anyone | 507.13 | 80 | 570 |
| 59 | Jackson LLC_report | 513.64 | 64 | 510 |
| 60 | Jackson, Collier and Barber_result | 1316.82 | 98 | 690 |
| 61 | Jackson, White and Brown_expect | 1409.21 | 102 | 760 |
| 62 | Jackson-Carroll_six | 1237.37 | 40 | 270 |
| 63 | Lee-Jones_soldier | 1042.66 | 110 | 880 |
| 64 | Lewis, Sanchez and Turner_point | 156.02 | 59 | 470 |
| 65 | Lin LLC_performance | 728.62 | 97 | 670 |
| 66 | Lindsey, Avila and Brown_candidate | 971.93 | 90 | 620 |
| 67 | Lindsey-Strickland_modern | 1128.07 | 43 | 310 |
| 68 | Little-Perez_choice | 256.42 | 96 | 670 |
| 69 | Lloyd, Stone and Mcguire_national | 1064.72 | 21 | 150 |
| 70 | Long, Hughes and Gallegos_receive | 348.43 | 55 | 440 |
| 71 | Lopez PLC_analysis | 1183.94 | 30 | 210 |

So, for each $i=1,\ldots,71$ (in the above order):

$0 \leq x_i \leq \min\{\text{Demand}_i,\, \text{Initial Inventory}_i\}$, integer

$\max \sum_{i=1}^{71} \text{Revenue}_i \cdot x_i$

where all coefficients and bounds are as above.