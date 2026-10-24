{
  "query": "A service team must cover eight jobs. The offer list gives cost_cents for each available worker_id/project_id pairing among ten staff members. Each project needs exactly one person, and each person may take at most one project. Exclude anyone with on_leave=1. The assigned person must meet required_skill, with Junior < Intermediate < Senior < Expert. Only listed offers are permitted. Minimize the sum of assignment costs, and report the minimum in USD cents.",
  "relationships": [],
  "tables": [
    {
      "columns": [
        "worker_id",
        "skill",
        "on_leave"
      ],
      "file_name": "export_01.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "on_leave": "1",
            "skill": "Senior",
            "worker_id": "W01"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "on_leave": "0",
            "skill": "Junior",
            "worker_id": "W00"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "on_leave": "0",
            "skill": "Junior",
            "worker_id": "W19"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "on_leave": "0",
            "skill": "Senior",
            "worker_id": "W11"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "on_leave": "0",
            "skill": "Junior",
            "worker_id": "W04"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "on_leave": "0",
            "skill": "Expert",
            "worker_id": "W15"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "on_leave": "0",
            "skill": "Expert",
            "worker_id": "W10"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "on_leave": "0",
            "skill": "Expert",
            "worker_id": "W06"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "on_leave": "0",
            "skill": "Intermediate",
            "worker_id": "W20"
          }
        },
        {
          "data_row_number": 9,
          "values": {
            "on_leave": "0",
            "skill": "Intermediate",
            "worker_id": "W14"
          }
        }
      ],
      "table_id": "file_0_view_0"
    },
    {
      "columns": [
        "project_id",
        "required_skill"
      ],
      "file_name": "export_02.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "project_id": "P00",
            "required_skill": "Intermediate"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "project_id": "P01",
            "required_skill": "Intermediate"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "project_id": "P02",
            "required_skill": "Senior"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "project_id": "P03",
            "required_skill": "Junior"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "project_id": "P04",
            "required_skill": "Junior"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "project_id": "P05",
            "required_skill": "Junior"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "project_id": "P06",
            "required_skill": "Junior"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "project_id": "P07",
            "required_skill": "Junior"
          }
        }
      ],
      "table_id": "file_1_view_0"
    },
    {
      "columns": [
        "worker_id",
        "project_id",
        "cost_cents"
      ],
      "file_name": "export_03.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "cost_cents": "147",
            "project_id": "P00",
            "worker_id": "W10"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "cost_cents": "114",
            "project_id": "P01",
            "worker_id": "W10"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "cost_cents": "998",
            "project_id": "P03",
            "worker_id": "W10"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "cost_cents": "1270",
            "project_id": "P04",
            "worker_id": "W10"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "cost_cents": "953",
            "project_id": "P07",
            "worker_id": "W10"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "cost_cents": "5",
            "project_id": "P00",
            "worker_id": "W19"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "cost_cents": "9",
            "project_id": "P02",
            "worker_id": "W19"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "cost_cents": "109",
            "project_id": "P03",
            "worker_id": "W19"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "cost_cents": "1306",
            "project_id": "P04",
            "worker_id": "W19"
          }
        },
        {
          "data_row_number": 9,
          "values": {
            "cost_cents": "626",
            "project_id": "P05",
            "worker_id": "W19"
          }
        },
        {
          "data_row_number": 10,
          "values": {
            "cost_cents": "466",
            "project_id": "P06",
            "worker_id": "W19"
          }
        },
        {
          "data_row_number": 11,
          "values": {
            "cost_cents": "1365",
            "project_id": "P07",
            "worker_id": "W19"
          }
        },
        {
          "data_row_number": 12,
          "values": {
            "cost_cents": "235",
            "project_id": "P00",
            "worker_id": "W11"
          }
        },
        {
          "data_row_number": 13,
          "values": {
            "cost_cents": "1034",
            "project_id": "P02",
            "worker_id": "W11"
          }
        },
        {
          "data_row_number": 14,
          "values": {
            "cost_cents": "782",
            "project_id": "P03",
            "worker_id": "W11"
          }
        },
        {
          "data_row_number": 15,
          "values": {
            "cost_cents": "556",
            "project_id": "P04",
            "worker_id": "W11"
          }
        },
        {
          "data_row_number": 16,
          "values": {
            "cost_cents": "1018",
            "project_id": "P05",
            "worker_id": "W11"
          }
        },
        {
          "data_row_number": 17,
          "values": {
            "cost_cents": "1136",
            "project_id": "P07",
            "worker_id": "W11"
          }
        },
        {
          "data_row_number": 18,
          "values": {
            "cost_cents": "17",
            "project_id": "P01",
            "worker_id": "W00"
          }
        },
        {
          "data_row_number": 19,
          "values": {
            "cost_cents": "430",
            "project_id": "P04",
            "worker_id": "W00"
          }
        },
        {
          "data_row_number": 20,
          "values": {
            "cost_cents": "1385",
            "project_id": "P05",
            "worker_id": "W00"
          }
        },
        {
          "data_row_number": 21,
          "values": {
            "cost_cents": "260",
            "project_id": "P06",
            "worker_id": "W00"
          }
        },
        {
          "data_row_number": 22,
          "values": {
            "cost_cents": "1107",
            "project_id": "P07",
            "worker_id": "W00"
          }
        },
        {
          "data_row_number": 23,
          "values": {
            "cost_cents": "675",
            "project_id": "P00",
            "worker_id": "W20"
          }
        },
        {
          "data_row_number": 24,
          "values": {
            "cost_cents": "1055",
            "project_id": "P01",
            "worker_id": "W20"
          }
        },
        {
          "data_row_number": 25,
          "values": {
            "cost_cents": "8",
            "project_id": "P02",
            "worker_id": "W20"
          }
        },
        {
          "data_row_number": 26,
          "values": {
            "cost_cents": "703",
            "project_id": "P03",
            "worker_id": "W20"
          }
        },
        {
          "data_row_number": 27,
          "values": {
            "cost_cents": "893",
            "project_id": "P06",
            "worker_id": "W20"
          }
        },
        {
          "data_row_number": 28,
          "values": {
            "cost_cents": "129",
            "project_id": "P07",
            "worker_id": "W20"
          }
        },
        {
          "data_row_number": 29,
          "values": {
            "cost_cents": "4",
            "project_id": "P00",
            "worker_id": "W04"
          }
        },
        {
          "data_row_number": 30,
          "values": {
            "cost_cents": "8",
            "project_id": "P01",
            "worker_id": "W04"
          }
        },
        {
          "data_row_number": 31,
          "values": {
            "cost_cents": "16",
            "project_id": "P02",
            "worker_id": "W04"
          }
        },
        {
          "data_row_number": 32,
          "values": {
            "cost_cents": "543",
            "project_id": "P03",
            "worker_id": "W04"
          }
        },
        {
          "data_row_number": 33,
          "values": {
            "cost_cents": "205",
            "project_id": "P04",
            "worker_id": "W04"
          }
        },
        {
          "data_row_number": 34,
          "values": {
            "cost_cents": "1149",
            "project_id": "P05",
            "worker_id": "W04"
          }
        },
        {
          "data_row_number": 35,
          "values": {
            "cost_cents": "533",
            "project_id": "P06",
            "worker_id": "W04"
          }
        },
        {
          "data_row_number": 36,
          "values": {
            "cost_cents": "732",
            "project_id": "P07",
            "worker_id": "W04"
          }
        },
        {
          "data_row_number": 37,
          "values": {
            "cost_cents": "19",
            "project_id": "P00",
            "worker_id": "W01"
          }
        },
        {
          "data_row_number": 38,
          "values": {
            "cost_cents": "6",
            "project_id": "P02",
            "worker_id": "W01"
          }
        },
        {
          "data_row_number": 39,
          "values": {
            "cost_cents": "19",
            "project_id": "P05",
            "worker_id": "W01"
          }
        },
        {
          "data_row_number": 40,
          "values": {
            "cost_cents": "8",
            "project_id": "P06",
            "worker_id": "W01"
          }
        },
        {
          "data_row_number": 41,
          "values": {
            "cost_cents": "13",
            "project_id": "P07",
            "worker_id": "W01"
          }
        },
        {
          "data_row_number": 42,
          "values": {
            "cost_cents": "1217",
            "project_id": "P00",
            "worker_id": "W06"
          }
        },
        {
          "data_row_number": 43,
          "values": {
            "cost_cents": "425",
            "project_id": "P01",
            "worker_id": "W06"
          }
        },
        {
          "data_row_number": 44,
          "values": {
            "cost_cents": "1320",
            "project_id": "P02",
            "worker_id": "W06"
          }
        },
        {
          "data_row_number": 45,
          "values": {
            "cost_cents": "1097",
            "project_id": "P03",
            "worker_id": "W06"
          }
        },
        {
          "data_row_number": 46,
          "values": {
            "cost_cents": "822",
            "project_id": "P04",
            "worker_id": "W06"
          }
        },
        {
          "data_row_number": 47,
          "values": {
            "cost_cents": "221",
            "project_id": "P05",
            "worker_id": "W06"
          }
        },
        {
          "data_row_number": 48,
          "values": {
            "cost_cents": "496",
            "project_id": "P06",
            "worker_id": "W06"
          }
        },
        {
          "data_row_number": 49,
          "values": {
            "cost_cents": "908",
            "project_id": "P07",
            "worker_id": "W06"
          }
        },
        {
          "data_row_number": 50,
          "values": {
            "cost_cents": "1361",
            "project_id": "P01",
            "worker_id": "W14"
          }
        },
        {
          "data_row_number": 51,
          "values": {
            "cost_cents": "379",
            "project_id": "P03",
            "worker_id": "W14"
          }
        },
        {
          "data_row_number": 52,
          "values": {
            "cost_cents": "239",
            "project_id": "P04",
            "worker_id": "W14"
          }
        },
        {
          "data_row_number": 53,
          "values": {
            "cost_cents": "460",
            "project_id": "P05",
            "worker_id": "W14"
          }
        },
        {
          "data_row_number": 54,
          "values": {
            "cost_cents": "130",
            "project_id": "P06",
            "worker_id": "W14"
          }
        },
        {
          "data_row_number": 55,
          "values": {
            "cost_cents": "722",
            "project_id": "P00",
            "worker_id": "W15"
          }
        },
        {
          "data_row_number": 56,
          "values": {
            "cost_cents": "577",
            "project_id": "P02",
            "worker_id": "W15"
          }
        },
        {
          "data_row_number": 57,
          "values": {
            "cost_cents": "1062",
            "project_id": "P03",
            "worker_id": "W15"
          }
        },
        {
          "data_row_number": 58,
          "values": {
            "cost_cents": "1314",
            "project_id": "P04",
            "worker_id": "W15"
          }
        },
        {
          "data_row_number": 59,
          "values": {
            "cost_cents": "528",
            "project_id": "P05",
            "worker_id": "W15"
          }
        },
        {
          "data_row_number": 60,
          "values": {
            "cost_cents": "835",
            "project_id": "P06",
            "worker_id": "W15"
          }
        },
        {
          "data_row_number": 61,
          "values": {
            "cost_cents": "918",
            "project_id": "P07",
            "worker_id": "W15"
          }
        }
      ],
      "table_id": "file_2_view_0"
    }
  ]
}
