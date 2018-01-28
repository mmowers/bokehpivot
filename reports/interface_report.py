import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import reeds_bokeh as rb
import importlib

report_path = sys.argv[1]
report_name = sys.argv[2]
path = sys.argv[3]
base = sys.argv[4]
html_num = sys.argv[5]

sys.path.insert(1, report_path)
report = importlib.import_module(report_name)
rb.reeds_static(path, report.static_presets, base=base, report_path= report_path, report_name= report_name, report_format='both', html_num=html_num)
