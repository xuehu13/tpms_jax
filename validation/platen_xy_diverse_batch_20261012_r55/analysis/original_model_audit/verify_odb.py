"""Read-only check of actual historical/current Abaqus databases; no solver job."""
import os, json
from odbAccess import openOdb

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
oldroot = r'F:\auto_abaqus\work'
native = r'E:\ABAQUS\2026temp\Abaqus_Work\tpms_jax_abaqus'
records = []
for kind in ['diverse_02', 'diverse_04', 'diverse_05', 'diverse_28']:
    parent = 'r3_smoke/batch_001' if kind == 'diverse_02' else 'para_aly/Fine/T0p02/MS0'
    old = os.path.join(oldroot, parent, kind, 'abaqus', kind + '_solve.odb')
    folder = 'r54_diverse04_midsurface_end_20pct_20261012' if kind == 'diverse_04' else 'r55_' + kind + '_midsurface_end_20pct_20261012'
    for label, path in [('original', old), ('current', os.path.join(native, folder, 'platen.odb'))]:
        if label == 'original':
            with open(os.path.join(oldroot,parent,kind,'abaqus','build_report.json')) as stream:
                labels = [json.load(stream)['model']['labels']['rp_top']]
        else:
            reading = os.path.join(W,'output','r54_platen_xy_diverse04_20261012') if kind == 'diverse_04' else os.path.join(HERE,'..',kind)
            with open(os.path.join(reading,'model_parameters.json')) as stream:
                labels = [json.load(stream)['labels']['top']]
        odb = openOdb(path=path, readOnly=True)
        try:
            step = list(odb.steps.values())[-1]
            regions = [(name, reg) for name, reg in step.historyRegions.items()
                       if 'U3' in reg.historyOutputs and 'RF3' in reg.historyOutputs
                       and any(name.split('.')[-1] == str(n) for n in labels)]
            assert len(regions) == 1, (path, labels, list(step.historyRegions.keys()))
            name, reg = regions[0]
            u = list(reg.historyOutputs['U3'].data)
            f = list(reg.historyOutputs['RF3'].data)
            loading = [(a[0], -a[1]/10., -b[1]/100.) for a,b in zip(u,f)
                       if a[0] <= (0.004000001 if label == 'current' else step.timePeriod + 1e-8)]
            peak = max(loading, key=lambda row: row[2])
            materials = {}
            for mname, mat in odb.materials.items():
                values = {}
                for prop in ['elastic', 'plastic', 'hyperelastic', 'density']:
                    obj = getattr(mat, prop, None)
                    if obj is not None:
                        try:
                            values[prop] = list(obj.table)
                        except Exception:
                            pass
                materials[mname] = values
            sections = {}
            for sname, sec in odb.sections.items():
                sections[sname] = {'type':type(sec).__name__, 'thickness':getattr(sec,'thickness',None)}
            records.append({'kind':kind,'label':label,'odb':path,'step':step.name,
                            'time_period_s':step.timePeriod,'top_labels':labels,'top_history':name,
                            'history_samples':len(u),'loading_raw_peak':[float(x) for x in peak],
                            'materials':materials,'sections':sections})
        finally:
            odb.close()
with open(os.path.join(HERE, 'odb_readonly_check.json'), 'w') as stream:
    json.dump(records,stream,ensure_ascii=True,indent=2)
print(json.dumps({'databases_read':len(records),'new_analysis_jobs':0,'peaks':[(r['kind'],r['label'],r['loading_raw_peak']) for r in records]}))
