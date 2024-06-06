import sif_parser

dir_path = r"\\PC160\GroupOdening\OM_data_Bern\raw_data\2024\20240327\Videosdata\experiment\pacing\50mmHg\6mmHg_baseline\20240327_13h-43m-40"
dir_path_local = r"C:\Users\lopez\Desktop\test_20240327\20240327_13h-40m-45"

_ = sif_parser.np_spool_open(dir_path_local, multithreading= True, max_workers=32)