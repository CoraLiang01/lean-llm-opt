##### Decision Variables

Let $x_i$ denote the integer quantity of orders fulfilled for product $i$, where $i$ indexes the products listed below.

##### Objective Function

$\max \sum_{i=1}^{70} r_i x_i$

where $r_i$ is the revenue per unit for product $i$.

##### Constraints

For each product $i$:

$0 \leq x_i \leq \min\{\text{Demand}_i, \text{Initial Inventory}_i\}$

$x_i \in \mathbb{Z}_{\geq 0}$

##### Retrieved Information

{
  "products": [
    {"Product Name": "Adams Group_service", "Revenue": 1003.56, "Demand": 10, "Initial Inventory": 80},
    {"Product Name": "Anderson-Leach_against", "Revenue": 549.92, "Demand": 6, "Initial Inventory": 40},
    {"Product Name": "Anderson-Valdez_somebody", "Revenue": 169.7, "Demand": 75, "Initial Inventory": 560},
    {"Product Name": "Anderson-White_son", "Revenue": 1285.81, "Demand": 121, "Initial Inventory": 820},
    {"Product Name": "Andrews LLC_matter", "Revenue": 1269.71, "Demand": 130, "Initial Inventory": 960},
    {"Product Name": "Andrews PLC_enter", "Revenue": 805.82, "Demand": 112, "Initial Inventory": 850},
    {"Product Name": "Andrews-Martin_build", "Revenue": 1134.19, "Demand": 125, "Initial Inventory": 950},
    {"Product Name": "Arias-Mendoza_life", "Revenue": 1479.23, "Demand": 54, "Initial Inventory": 360},
    {"Product Name": "Bennett and Sons_down", "Revenue": 139.14, "Demand": 75, "Initial Inventory": 580},
    {"Product Name": "Bennett, Foster and Moreno_enter", "Revenue": 518.67, "Demand": 102, "Initial Inventory": 800},
    {"Product Name": "Fernandez, Long and Nelson_member", "Revenue": 1088.57, "Demand": 48, "Initial Inventory": 340},
    {"Product Name": "Fernandez-Fischer_million", "Revenue": 371.56, "Demand": 4, "Initial Inventory": 30},
    {"Product Name": "Ferrell Inc_prove", "Revenue": 956.78, "Demand": 125, "Initial Inventory": 990},
    {"Product Name": "Fields, Christensen and Daniels_nor", "Revenue": 1283.7, "Demand": 9, "Initial Inventory": 70},
    {"Product Name": "Figueroa LLC_involve", "Revenue": 1184.54, "Demand": 132, "Initial Inventory": 890},
    {"Product Name": "Finley Group_itself", "Revenue": 377.8, "Demand": 29, "Initial Inventory": 240},
    {"Product Name": "Fisher Ltd_speak", "Revenue": 701.97, "Demand": 43, "Initial Inventory": 290},
    {"Product Name": "Fisher-Marshall_cover", "Revenue": 875.91, "Demand": 5, "Initial Inventory": 40},
    {"Product Name": "Fuller-Torres_behavior", "Revenue": 951.85, "Demand": 40, "Initial Inventory": 300},
    {"Product Name": "Fuller-Walters_radio", "Revenue": 869.95, "Demand": 48, "Initial Inventory": 350},
    {"Product Name": "Gallagher-Campbell_dark", "Revenue": 887.06, "Demand": 87, "Initial Inventory": 600},
    {"Product Name": "Gallagher-Kirby_talk", "Revenue": 1318.33, "Demand": 79, "Initial Inventory": 610},
    {"Product Name": "Gallagher-Parker_recognize", "Revenue": 448.61, "Demand": 20, "Initial Inventory": 160},
    {"Product Name": "Gonzalez PLC_task", "Revenue": 167.99, "Demand": 33, "Initial Inventory": 260},
    {"Product Name": "Gonzalez, Coleman and Le_heavy", "Revenue": 1068.1, "Demand": 60, "Initial Inventory": 480},
    {"Product Name": "Gonzalez, Lowe and Robinson_education", "Revenue": 872.16, "Demand": 25, "Initial Inventory": 200},
    {"Product Name": "Gonzalez-Horn_light", "Revenue": 483.26, "Demand": 42, "Initial Inventory": 290},
    {"Product Name": "Good, Davis and Smith_station", "Revenue": 1279.24, "Demand": 52, "Initial Inventory": 350},
    {"Product Name": "Goodman, Hughes and White_realize", "Revenue": 894.34, "Demand": 116, "Initial Inventory": 900},
    {"Product Name": "Goodwin PLC_bad", "Revenue": 367.91, "Demand": 95, "Initial Inventory": 760},
    {"Product Name": "Gordon PLC_detail", "Revenue": 178.22, "Demand": 66, "Initial Inventory": 440},
    {"Product Name": "Graham LLC_stage", "Revenue": 1030.8, "Demand": 25, "Initial Inventory": 170},
    {"Product Name": "Graham Ltd_marriage", "Revenue": 479.44, "Demand": 78, "Initial Inventory": 600},
    {"Product Name": "Graham-Swanson_message", "Revenue": 973.62, "Demand": 138, "Initial Inventory": 920},
    {"Product Name": "Grant, Cross and Bennett_religious", "Revenue": 1211.35, "Demand": 32, "Initial Inventory": 240},
    {"Product Name": "Grant, Mcdonald and Watson_owner", "Revenue": 467.93, "Demand": 28, "Initial Inventory": 190},
    {"Product Name": "Graves, Turner and Crawford_wait", "Revenue": 678.72, "Demand": 84, "Initial Inventory": 600},
    {"Product Name": "Gray, Smith and Barnes_much", "Revenue": 478.36, "Demand": 102, "Initial Inventory": 760},
    {"Product Name": "Green Inc_direction", "Revenue": 1196.95, "Demand": 112, "Initial Inventory": 850},
    {"Product Name": "Green-Rogers_could", "Revenue": 1118.63, "Demand": 7, "Initial Inventory": 50},
    {"Product Name": "Greene-Baxter_them", "Revenue": 972.1, "Demand": 62, "Initial Inventory": 470},
    {"Product Name": "Greer and Sons_keep", "Revenue": 338.6, "Demand": 109, "Initial Inventory": 790},
    {"Product Name": "Griffin, Boyle and Dawson_anything", "Revenue": 178.52, "Demand": 81, "Initial Inventory": 580},
    {"Product Name": "Harper Inc_well", "Revenue": 992.14, "Demand": 131, "Initial Inventory": 880},
    {"Product Name": "Harper Ltd_off", "Revenue": 498.77, "Demand": 26, "Initial Inventory": 170},
    {"Product Name": "Harrington, Sosa and Mccarty_coach", "Revenue": 965.34, "Demand": 109, "Initial Inventory": 790},
    {"Product Name": "Harris Group_different", "Revenue": 321.11, "Demand": 65, "Initial Inventory": 500},
    {"Product Name": "Harris and Sons_audience", "Revenue": 287.36, "Demand": 121, "Initial Inventory": 890},
    {"Product Name": "Harris and Sons_fear", "Revenue": 462.26, "Demand": 90, "Initial Inventory": 680},
    {"Product Name": "Harris, Hamilton and Rose_contain", "Revenue": 641.22, "Demand": 31, "Initial Inventory": 250},
    {"Product Name": "Harris, Stevens and Hall_answer", "Revenue": 1136.87, "Demand": 24, "Initial Inventory": 170},
    {"Product Name": "Harris-Bell_painting", "Revenue": 1385.88, "Demand": 122, "Initial Inventory": 830},
    {"Product Name": "Harris-Melton_often", "Revenue": 867.3, "Demand": 101, "Initial Inventory": 680},
    {"Product Name": "Harris-Rogers_after", "Revenue": 1175.92, "Demand": 76, "Initial Inventory": 590},
    {"Product Name": "Hart Group_cold", "Revenue": 153.61, "Demand": 100, "Initial Inventory": 700},
    {"Product Name": "Hutchinson, Roberts and Mcbride_among", "Revenue": 753.55, "Demand": 125, "Initial Inventory": 830},
    {"Product Name": "Ibarra, Jackson and Potter_upon", "Revenue": 245.81, "Demand": 45, "Initial Inventory": 360},
    {"Product Name": "Jackson Inc_anyone", "Revenue": 507.13, "Demand": 80, "Initial Inventory": 570},
    {"Product Name": "Jackson LLC_report", "Revenue": 513.64, "Demand": 64, "Initial Inventory": 510},
    {"Product Name": "Jackson, Collier and Barber_result", "Revenue": 1316.82, "Demand": 98, "Initial Inventory": 690},
    {"Product Name": "Jackson, White and Brown_expect", "Revenue": 1409.21, "Demand": 102, "Initial Inventory": 760},
    {"Product Name": "Jackson-Carroll_six", "Revenue": 1237.37, "Demand": 40, "Initial Inventory": 270},
    {"Product Name": "Lee-Jones_soldier", "Revenue": 1042.66, "Demand": 110, "Initial Inventory": 880},
    {"Product Name": "Lewis, Sanchez and Turner_point", "Revenue": 156.02, "Demand": 59, "Initial Inventory": 470},
    {"Product Name": "Lin LLC_performance", "Revenue": 728.62, "Demand": 97, "Initial Inventory": 670},
    {"Product Name": "Lindsey, Avila and Brown_candidate", "Revenue": 971.93, "Demand": 90, "Initial Inventory": 620},
    {"Product Name": "Lindsey-Strickland_modern", "Revenue": 1128.07, "Demand": 43, "Initial Inventory": 310},
    {"Product Name": "Little-Perez_choice", "Revenue": 256.42, "Demand": 96, "Initial Inventory": 670},
    {"Product Name": "Lloyd, Stone and Mcguire_national", "Revenue": 1064.72, "Demand": 21, "Initial Inventory": 150},
    {"Product Name": "Long, Hughes and Gallegos_receive", "Revenue": 348.43, "Demand": 55, "Initial Inventory": 440},
    {"Product Name": "Lopez PLC_analysis", "Revenue": 1183.94, "Demand": 30, "Initial Inventory": 210}
  ]
}

##### Full Model

Let $i$ index the products in the list above.

$\max \sum_{i} r_i x_i$

subject to

$0 \leq x_i \leq \min\{\text{Demand}_i, \text{Initial Inventory}_i\}$

$x_i \in \mathbb{Z}_{\geq 0}$

where for each product $i$:

- $r_i$ = Revenue per unit (see list above)
- $\text{Demand}_i$ = Demand (see list above)
- $\text{Initial Inventory}_i$ = Initial Inventory (see list above)
- $x_i$ = quantity of orders fulfilled for product $i$