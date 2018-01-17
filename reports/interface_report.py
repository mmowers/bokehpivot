import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], 'templates'))
import reeds_bokeh as rb
import importlib
report = importlib.import_module(sys.argv[1])

path = sys.argv[2]
base = sys.argv[3]
html_num = sys.argv[4]
rb.reeds_static(path, report.static_presets, base=base, report_format='both', html_num=html_num)
