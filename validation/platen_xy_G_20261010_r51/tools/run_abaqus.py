"""Windows launcher for this one isolated G cut-end experiment."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,time
D=Path(__file__).resolve().parents[1]
NATIVE=Path(r'E:\ABAQUS\2026temp\Abaqus_Work\tpms_jax_abaqus\platen_xy_G_20261010_r51_2pct')
ABQ=r'E:\ABAQUS\2026\Commands\abaqus.bat'
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=['datacheck','solve','extract']);a.add_argument('--native-dir',type=Path);a.add_argument('--output-dir',type=Path);a.add_argument('--input',type=Path);a.add_argument('--config',type=Path);a.add_argument('--extract-script',type=Path);x=a.parse_args()
    native=x.native_dir or NATIVE;out=x.output_dir or D/'abaqus'
    native.mkdir(exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    if x.action=='datacheck':
        if (native/'check.dat').exists():raise FileExistsError('Do not repeat datacheck in original directory')
        shutil.copy2(x.input or D/'abaqus/platen_2pct.inp',native/'platen.inp');shutil.copy2(x.config or D/'input/protocol.json',native/'protocol.json')
    if x.action=='solve':
        check=json.loads((out/'datacheck_receipt.json').read_text());assert check['success']
        if (native/'platen.odb').exists():raise FileExistsError('Never overwrite a solver attempt')
    if x.action=='extract':
        shutil.copy2(x.extract_script or D/'tools/extract_abaqus.py',native/'extract.py')
        cmd=['cmd','/d','/c',ABQ,'python','extract.py','platen.odb','protocol.json','result.json'];limit=120
    else:
        job='check' if x.action=='datacheck' else 'platen'
        cmd=['cmd','/d','/c',ABQ,'job='+job,'input=platen.inp','cpus=4','double=both','interactive','ask_delete=OFF']
        if x.action=='datacheck':cmd.append('datacheck')
        limit=180 if x.action=='datacheck' else 1450
    start=time.monotonic();stopped=False
    with (native/(x.action+'_launch.log')).open('w') as log:
        p=subprocess.Popen(cmd,cwd=native,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        try:code=p.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
            code=p.wait();stopped=True
    success=code==0 and not stopped
    if x.action=='extract':
        # Abaqus python may return 0 even when the script raises an exception.
        success=success and (native/'result.json').exists() and (native/'field.npz').exists()
        if success:success=json.loads((native/'result.json').read_text())['complete']
    if x.action in ['datacheck','solve']:
        dat=(native/('check.dat' if x.action=='datacheck' else 'platen.dat')).read_text(errors='replace')
        success=success and '***ERROR' not in dat
        if x.action=='solve':success=success and 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in (native/'platen.sta').read_text(errors='replace')
    receipt={'action':x.action,'command':cmd,'returncode':code,'wall_seconds':time.monotonic()-start,'budget_stop':stopped,
             'success':success,'native_directory':str(native),'INP_sha256':hashlib.sha256((native/'platen.inp').read_bytes()).hexdigest()}
    write(out/(x.action+'_receipt.json'),receipt);write(native/(x.action+'_receipt.json'),receipt)
    if x.action=='extract' and success:shutil.copy2(native/'result.json',out/'result.json');shutil.copy2(native/'field.npz',out/'field.npz')
    print(json.dumps(receipt,indent=2))
    if not success:raise SystemExit(1)
if __name__=='__main__':main()
