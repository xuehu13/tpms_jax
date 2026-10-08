"""Record datacheck outcome using the actual Abaqus log, not dispatcher rc."""
from pathlib import Path
import hashlib,json,re,shutil
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/shell_fill_diagnostic_20261007_r25'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
P=Path('/mnt/e/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/shell_fill_20261007_r25_nearzero')
phase=__import__('sys').argv[1]
job='check_gauge' if phase=='datacheck_gauge' else ('check_serial' if phase=='datacheck_serial' else 'shell_fill')
inpname='shell_fill_gauge.inp' if phase=='datacheck_gauge' else 'shell_fill.inp'
log=(P/(phase+'.log')).read_text(errors='replace');dat=(P/(job+'.dat')).read_text(errors='replace')
errors=re.findall(r'\*\*\*ERROR[^\n]*',dat,re.I)
warnings=re.findall(r'\*\*\*WARNING[^\n]*(?:\n[^\n]*){0,3}',dat,re.I)
# Parse the printed embedding table (weights have limited print precision).
tail=dat.split('  EMBEDDED   NUMBER OF')[1]
rows={};current=None
for line in tail.splitlines()[2:]:
    t=line.split()
    if not t:continue
    try:
        if len(t)>=4 and 1<=int(t[0])<=8372 and int(t[1]) in range(1,9) and int(t[2])>=100001:
            current=int(t[0]);rows[current]={'n':int(t[1]),'tokens':t[2:]}
        elif current is not None and int(t[0])>=100001:rows[current]['tokens']+=t
        elif current is not None:break
    except ValueError:
        if current is not None:break
maxsum=0.;weightmin=1.;countok=True
for v in rows.values():
    x=v['tokens'];countok &= len(x)==2*v['n']
    weights=[float(x[i]) for i in range(1,len(x),2)]
    maxsum=max(maxsum,abs(sum(weights)-1));weightmin=min(weightmin,min(weights))
success=('Abaqus JOB '+job+' COMPLETED') in log and not errors and 'exited with errors' not in log.lower() and 'exception' not in log.lower()
cfg=json.loads((P/'input.json').read_text())
gate={'phase':phase,'Abaqus_log_success':success,'dispatcher_returncode_not_a_success_test':True,
    'system_exception':'EXCEPTION_ACCESS_VIOLATION' in log,'input_errors':errors,'warnings':warnings,
    'embedded_shell_nodes':len(rows),'all_8372_embedded':len(rows)==8372,
    'host_weight_counts_ok':bool(countok),'max_printed_weight_sum_error':maxsum,
    'printed_weight_min':weightmin,'printed_precision_not_exact_interpolation_verification':True,
    'approved_for_nearzero_solve':bool(success and len(rows)==8372 and countok and weightmin>=0 and maxsum<1e-4),
    'INP_sha256':hashlib.sha256((P/inpname).read_bytes()).hexdigest()}
out=D/'datacheck';out.mkdir(exist_ok=True)
for name in [phase+'.log',phase+'_receipt.json',job+'.dat',job+'.sta']:
    if (P/name).exists():shutil.copy2(P/name,out/name)
(out/(phase+'_gate.json')).write_text(json.dumps(gate,indent=2))
if phase in ['datacheck_serial','datacheck_gauge'] and gate['approved_for_nearzero_solve']:
    name='datacheck_gauge_gate.json' if phase=='datacheck_gauge' else 'datacheck_gate.json'
    (P/name).write_text(json.dumps(gate,indent=2))
print(json.dumps(gate,indent=2))
