from pathlib import Path
import os, runpy, sys
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
script=(root/sys.argv[1]).resolve() if len(sys.argv)>1 else root/"verify_package.py"
if not script.is_relative_to(root) or not script.is_file():raise SystemExit("Choose an existing script inside this package")
sys.argv=[str(script),*sys.argv[2:]] if len(sys.argv)>1 else [str(script)]
os.chdir(script.parent if script.parent.name=="e_draw_plot" else root)
runpy.run_path(str(script),run_name="__main__")
