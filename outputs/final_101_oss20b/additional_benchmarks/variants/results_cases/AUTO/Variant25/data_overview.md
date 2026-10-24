{
  "query": "Assign six construction projects to the eight listed managers. Each project needs exactly one person, and each person may take at most one project. Exclude anyone with on_leave=1. The assigned person must meet required_skill, with Junior < Intermediate < Senior < Expert. In the cost matrix, project IDs are columns and worker IDs identify rows. A blank cell forbids that assignment. Minimize the sum of assignment costs, and report the minimum in USD cents.",
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
            "on_leave": "0",
            "skill": "Junior",
            "worker_id": "W12"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "on_leave": "0",
            "skill": "Expert",
            "worker_id": "W06"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "on_leave": "0",
            "skill": "Intermediate",
            "worker_id": "W04"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "on_leave": "1",
            "skill": "Junior",
            "worker_id": "W01"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "on_leave": "0",
            "skill": "Junior",
            "worker_id": "W11"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "on_leave": "0",
            "skill": "Intermediate",
            "worker_id": "W10"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "on_leave": "0",
            "skill": "Intermediate",
            "worker_id": "W02"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "on_leave": "0",
            "skill": "Expert",
            "worker_id": "W00"
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
            "required_skill": "Junior"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "project_id": "P01",
            "required_skill": "Junior"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "project_id": "P02",
            "required_skill": "Junior"
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
            "required_skill": "Intermediate"
          }
        }
      ],
      "table_id": "file_1_view_0"
    },
    {
      "columns": [
        "worker_id",
        "P00",
        "P01",
        "P02",
        "P03",
        "P04",
        "P05"
      ],
      "file_name": "export_03.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "P00": "1091",
            "P01": "324",
            "P02": "379",
            "P03": "272",
            "P04": "",
            "P05": "",
            "worker_id": "W11"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "P00": "1176",
            "P01": "1111",
            "P02": "1380",
            "P03": "542",
            "P04": "158",
            "P05": "922",
            "worker_id": "W10"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "P00": "",
            "P01": "822",
            "P02": "",
            "P03": "223",
            "P04": "1155",
            "P05": "1055",
            "worker_id": "W06"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "P00": "5",
            "P01": "",
            "P02": "20",
            "P03": "",
            "P04": "18",
            "P05": "7",
            "worker_id": "W01"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "P00": "",
            "P01": "1397",
            "P02": "953",
            "P03": "714",
            "P04": "205",
            "P05": "",
            "worker_id": "W02"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "P00": "1063",
            "P01": "219",
            "P02": "",
            "P03": "1329",
            "P04": "",
            "P05": "436",
            "worker_id": "W00"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "P00": "642",
            "P01": "",
            "P02": "1130",
            "P03": "133",
            "P04": "199",
            "P05": "311",
            "worker_id": "W04"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "P00": "102",
            "P01": "353",
            "P02": "651",
            "P03": "102",
            "P04": "",
            "P05": "7",
            "worker_id": "W12"
          }
        }
      ],
      "table_id": "file_2_view_0"
    }
  ]
}
