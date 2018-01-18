import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], 'templates'))
import reeds_bokeh as rb
import importlib

report_name = sys.argv[1]
path = sys.argv[2]
base = sys.argv[3]
html_num = sys.argv[4]
report = importlib.import_module(report_name)
rb.reeds_static(path, report.static_presets, base=base, report_name= report_name, report_format='both', html_num=html_num)
