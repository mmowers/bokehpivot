'''
Static HTML report maker

To use, modify the script below and run this file:
  * Add import statements for the desired report(s) from the templates folder
  * Enter the desired path variable and base variable, if required
  * Build the static_presets variable with the imported report or reports.
  * Add any other custom configurations using example_reeds_custom_config.py as a guide.
  * Build the appropriate call to rb.reeds_static
  * Save this file, open a command prompt and navigate to this folder, and run with "python create_report.py"

If running on Orion within the common bokehpivot folder, copy this file and edit/run the copy, rather than this file directly
'''
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
sys.path.insert(1, os.path.join(sys.path[0], '../templates'))
import reeds_bokeh as rb
import standard_compare_report

path = r'\\nrelqnap01d\ReEDS\FY17-JEDI-MRM-jedi\runs\2017-08-22 runs'
base = 'Base'
static_presets = standard_compare_report.static_presets

rb.reeds_static(path, static_presets, base=base, report_format='both', html_num='one')
'''
Notes on the function call:
    - core.static_report() may be called instead of reeds_bokeh.reeds_static(),
      but this disallows any static presets that use the top-level 'result',
      'preset', or 'modify' keys, which are reeds-specific, and 'base' is also
      not an argument to core.static_report().
    - 'report_format' allows 'excel', 'html', or 'both'
    - base is necessary if 'modify' is used in any of the presets
'''