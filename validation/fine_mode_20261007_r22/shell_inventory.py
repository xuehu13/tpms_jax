from pathlib import Path
import json,numpy as np,hashlib,re
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/fine_mode_20261007_r22'
S=R/'validation/shell_rate_20261007_r20/abaqus/explicit_T0p004';F=S/'results/shell_frames'
rows=json.loads((F/'modes.json').read_text())['rows'];row=min(rows,key=lambda x:abs(x['compression']-.1186))
with np.load(F/row['file']) as a:keys=a.files
inp=(S/'thin_shell.inp').read_text();extract=(S/'extract_shell_modes.py').read_text()
sections=re.findall(r'\*Node Output[^\n]*\n([^*]+)',inp,flags=re.I)
result={'scope':'Only existing r20 data inventory; no ODB extraction, new job, mode solve or mechanical evaluation.',
 'example_frame':row['file'],'saved_frame_arrays':keys,'INP_node_output_sections':sections,
 'actual_frame_field_outputs_recorded_in_r20':row.get('field_outputs_available'),
 'extract_script_lines_mentioning_fieldOutputs_or_UR':[x for x in extract.splitlines() if 'fieldOutputs' in x or "'UR'" in x or '"UR"' in x],
 'file_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [S/'thin_shell.inp',S/'extract_shell_modes.py',F/row['file']]}}
(D/'shell_data_inventory.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
