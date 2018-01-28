import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import reeds_bokeh as rb
import importlib

data_source = sys.argv[1]
base = sys.argv[2]
report_name = sys.argv[3]
report_path = sys.argv[4]
html_num = sys.argv[5]
output_dir = sys.argv[6]
auto_open = sys.argv[7]

sys.path.insert(1, report_path)
report = importlib.import_module(report_name)
rb.reeds_static(data_source, base, report.static_presets, report_name, report_path, 'both', html_num, output_dir, auto_open)
