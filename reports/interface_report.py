import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], 'templates'))
import reeds_bokeh as rb
import importlib
report = importlib.import_module(sys.argv[1])

path = sys.argv[2]
if len(sys.argv) > 3:
  base = sys.argv[3]
  rb.reeds_static(path, report.static_presets, base, 'both')
else:
  rb.reeds_static(path, report.static_presets, report_format='both')
