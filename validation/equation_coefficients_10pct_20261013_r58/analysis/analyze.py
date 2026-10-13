"""One-off CPU summary for r58. No FEM, JAX, AD, FD, scheduler or navigation edits."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time

R = Path('/home/xuehu/projects/tpms_jax')
S = R / 'validation/equation_coefficients_20261012_r57'
B = R / 'validation/equation_coefficients_10pct_20261013_r58'
W = Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
O = W / 'output/r58_equation_coefficients_10pct_20261013'
A = B / 'analysis'
NAMES = ['c1', 'c2', 'c0']
PERTURBATIONS = [('plus_h', 1.), ('minus_h', -1.), ('plus_half', .5), ('minus_half', -.5)]
TERMINAL_STAGES = {'completed', 'completed_not_passed', 'stopped'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.analysis-tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def close(a, b, atol=1e-12, rtol=1e-12):
    return finite(a) and finite(b) and abs(a-b) <= atol + rtol*abs(b)


def gate(ad, coarse, fine, scale, objective_scale=1.):
    """Frozen 1% gate; near zero uses scaled absolute errors, never percentages."""
    if not all(finite(x) for x in [coarse, fine, scale, objective_scale]) or scale <= 0 or objective_scale <= 0:
        return dict(near_zero=None, AD_FD_error=None, FD_halving_error=None, FD_halving_passed=False, passed=False if ad is not None else None)
    near = abs(fine)*scale/objective_scale < 1e-6
    factor = scale/objective_scale if near else 1/abs(fine)
    fe = abs(coarse-fine)*factor
    ae = abs(ad-fine)*factor if finite(ad) else None
    limit = 1e-5 if near else .01
    return dict(near_zero=near, AD_FD_error=ae, FD_halving_error=fe,
                applicable_limit=limit, error_type='scaled_absolute' if near else 'relative',
                FD_halving_passed=fe <= limit, passed=(ae <= limit and fe <= limit) if ae is not None else None)


def process_alive(pid):
    # Only observes an explicitly supplied existing worker PID; never sends a signal.
    proc = Path('/proc') / str(pid)
    if not proc.exists():
        return False
    try:
        return (proc/'stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False


def wait_worker(pid, protocol):
    if pid <= 0 or pid == os.getpid():
        raise ValueError('--wait-pid must name the existing scientific worker, not this summary process')
    deadline = protocol['started_unix'] + protocol['total_wall_budget_s'] + 900
    while process_alive(pid) and time.time() < deadline:
        time.sleep(min(10., max(.01, deadline-time.time())))
    if process_alive(pid):
        raise RuntimeError('Worker remains alive beyond protocol deadline +900s; no final summary asserted')


def path_check(result, execution, prefix, prefix_source, parameters=None):
    reasons = []
    if not result:
        return dict(available=False, passed=False, reasons=['result.json not available'])
    if not result.get('complete'):
        reasons.append('complete flag is not true')
    if result.get('prefix_reused') is not True or result.get('prefix_source') != str(prefix_source):
        reasons.append('prefix provenance differs')
    rows = result.get('path', [])
    oldrows = (prefix or {}).get('path', [])
    if not oldrows or rows[:len(oldrows)] != oldrows:
        reasons.append('reused prefix rows differ from protected r57 path')
    if parameters is not None:
        actual = result.get('parameters', [])
        if len(actual) != 3 or not all(close(x, y, 1e-14, 1e-14) for x, y in zip(actual, parameters)):
            reasons.append('FD parameters differ from frozen perturbation')
    blocks = execution.get('prefix_blocks', []) + execution.get('extension_schedule', [])
    expected_counts = [x['count'] for x in blocks]
    if len(rows) != len(blocks):
        reasons.append('path row count differs from complete block schedule')
    cumulative_steps, cumulative_time = 0, 0.
    for i, (row, block) in enumerate(zip(rows, blocks)):
        cumulative_steps += block['count']
        cumulative_time += block['count']*block['dt_s']
        if row.get('steps') != cumulative_steps or not close(row.get('time_s'), cumulative_time):
            reasons.append(f'row {i}: steps/time differ from fixed block schedule')
            break
        if not all(finite(row.get(k)) for k in ['compression', 'work', 'force', 'min_required_J', 'min_actual_J', 'KE', 'IE']):
            reasons.append(f'row {i}: missing or nonfinite sampled response/diagnostics')
            break
        if row['min_required_J'] <= 0:
            reasons.append(f'row {i}: nonpositive sampled required-domain J')
            break
    terminal = result.get('terminal', {})
    if not rows or terminal != rows[-1]:
        reasons.append('terminal is not the last path row')
    if terminal.get('steps') != execution['steps'] or not close(terminal.get('time_s'), execution['end_time_s']) or not close(terminal.get('compression'), .1, 1e-10):
        reasons.append('terminal does not reach frozen steps/2ms/10%')
    if 'counts' in result and result['counts'] != expected_counts:
        reasons.append('JVP counts differ from full prefix plus extension schedule')
    return dict(available=True, passed=not reasons, reasons=reasons, complete_flag=result.get('complete'),
                terminal_steps=terminal.get('steps'), terminal_time_s=terminal.get('time_s'),
                terminal_compression=terminal.get('compression'), rows=len(rows), prefix_rows=len(oldrows))


def stored_check_matches(stored, recalculated):
    return bool(stored and recalculated and stored.get('passed') == recalculated.get('passed')
                and stored.get('near_zero') == recalculated.get('near_zero')
                and all(close(stored.get(k), recalculated.get(k), 1e-12, 1e-9)
                        for k in ['AD_FD_error', 'FD_halving_error']))


def plots(jvp, fd, work, checks, prefix_compression):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 180})
    files = ['response.png', 'trajectory_gradients.png', 'work_gradient_checks.png']
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    rows = (jvp or {}).get('path', [])
    if rows:
        x = [100*r['compression'] for r in rows]
        for axis, key, unit in zip(ax, ['force', 'work'], ['Force (N)', 'Loading work (N mm)']):
            axis.plot(x, [r[key] for r in rows], label='Baseline from c2 JVP')
            axis.axvline(prefix_compression*100, color='gray', ls=':', label='Reused prefix endpoint')
            axis.set(xlabel='Compression (%)', ylabel=unit)
            axis.legend(fontsize=8)
    else:
        for axis in ax: axis.text(.5, .5, 'Complete baseline path not available', ha='center', transform=axis.transAxes)
    fig.suptitle('r58: fixed 0-10% loading path; sampled responses')
    fig.savefig(A/files[0]); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for i, obj in enumerate(['work', 'force']):
        if rows and all(len(r.get('gradient', [])) == 2 for r in rows):
            ax[i].plot([r['compression']*100 for r in rows], [r['gradient'][i] for r in rows], label='c2 full JVP', lw=2)
        for name in NAMES:
            plus, minus = fd.get(name+'_plus_half'), fd.get(name+'_minus_half')
            if not plus or not minus or not checks[name+'_plus_half']['passed'] or not checks[name+'_minus_half']['passed']:
                continue
            pp, mm = plus['path'], minus['path']
            step = plus['parameters'][NAMES.index(name)]-minus['parameters'][NAMES.index(name)]
            if step and len(pp) == len(mm) and all(a['steps'] == b['steps'] and close(a['time_s'], b['time_s']) for a, b in zip(pp, mm)):
                ax[i].plot([r['compression']*100 for r in pp], [(a[obj]-b[obj])/step for a, b in zip(pp, mm)], ls='--', label=name+' fine FD')
        ax[i].axvline(prefix_compression*100, color='gray', ls=':')
        ax[i].set(xlabel='Compression (%)', ylabel='d'+('W' if obj == 'work' else 'F')+'/dc ('+('N mm' if obj == 'work' else 'N')+')')
        if ax[i].get_legend_handles_labels()[0]: ax[i].legend(fontsize=8)
        else: ax[i].text(.5, .5, 'Gradient paths pending', ha='center', transform=ax[i].transAxes)
    fig.suptitle('Trajectory sensitivities: c2 AD only; c1/c0 FD only')
    fig.savefig(A/files[1]); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    if work:
        x = list(range(len(work)))
        for offset, key, label in [(-.23, 'AD', 'Full VJP'), (0., 'FD_h', 'FD h'), (.23, 'FD_half', 'FD h/2')]:
            available = [(j, r[key]) for j, r in enumerate(work) if r[key] is not None]
            ax[0].bar([j+offset for j, _ in available], [v for _, v in available], .23, label=label)
        for offset, key, label in [(-.16, 'AD_FD_error', 'VJP vs fine FD'), (.16, 'FD_halving_error', 'FD halving')]:
            available = [(j, max(r[key]/r['applicable_limit'], 1e-12)) for j, r in enumerate(work) if r.get(key) is not None]
            ax[1].bar([j+offset for j, _ in available], [v for _, v in available], .32, label=label)
        for axis in ax: axis.set(xticks=x, xticklabels=[r['coefficient'] for r in work]); axis.legend(fontsize=8)
        ax[1].axhline(1, color='red', ls='--'); ax[1].set(yscale='log', ylabel='Error / frozen applicable gate')
    else:
        for axis in ax: axis.text(.5, .5, 'Full-path gradient checks pending', ha='center', transform=axis.transAxes)
    ax[0].set(ylabel='dW/dc (N mm / coefficient)')
    fig.suptitle('r58 loading-work checks; near-zero errors are scaled absolute')
    fig.savefig(A/files[2]); plt.close(fig)
    return files


def main(wait_pid=None):
    protocol = read(B/'input/protocol.json')
    if protocol is None:
        raise RuntimeError('r58 protocol not present; summary cannot infer or start an experiment')
    if wait_pid is not None: wait_worker(wait_pid, protocol)
    execution = read(B/'input/execution.json')
    if execution is None: raise RuntimeError('r58 frozen execution input is missing')
    A.mkdir(exist_ok=True)
    status = read(B/'status.json') or {}
    worker_status = dict(status)
    ended = status.get('stage') in TERMINAL_STAGES and not status.get('active_jobs')
    jvp = read(B/'AD/c2/result.json')
    vjp = read(B/'AD/work_VJP_result.json')
    prefix = read(S/'AD/c2/result.json')
    oldexecution = read(S/'input/execution.json')
    geo = read(S/'checks/geometry.json')
    oldmetrics = read(S/'analysis/metrics.json')
    bench = read(B/'checks/benchmark.json')
    equivalence = read(B/'checks/prefix_equivalence.json')
    stored_jvp = read(B/'checks/full_JVP_c2.json')
    stored_halving = read(B/'checks/work_FD_halving.json')
    failure = read(B/'failure.json')
    p, scales = protocol['baseline'], protocol['parameter_scales']
    h = [s*protocol['relative_fd_step'] for s in scales]
    thresholds = protocol.get('thresholds', {})
    frozen_protocol = bool(protocol['parameters'] == NAMES and p == [-3.4, 4.6, 2.] and scales == [3.4, 4.6, 2.]
        and protocol['relative_fd_step'] == 1e-5 and protocol['total_wall_budget_s'] == 28800
        and thresholds.get('relative_AD_FD') == .01 and thresholds.get('relative_FD_halving') == .01
        and thresholds.get('nearzero_scaled_cutoff') == 1e-6 and thresholds.get('nearzero_scaled_absolute_error') == 1e-5
        and thresholds.get('objective_scales') == [1., 1.] and execution['compression'] == .1 and execution['end_time_s'] == .002
        and protocol.get('target_compression') == .1 and protocol.get('target_time_s') == .002
        and execution['prefix_steps'] == 10113 and execution['steps'] == 34485 and execution.get('extension_steps') == 24372
        and oldexecution and execution['dt_s'] == oldexecution['dt_s']
        and execution.get('source_prefix') == str(S) and prefix
        and close(execution.get('prefix_time_s'), prefix['terminal']['time_s'])
        and [x['count'] for x in execution.get('prefix_blocks', [])] == prefix['counts'])
    fd, path_checks = {}, {}
    path_checks['JVP_c2'] = path_check(jvp, execution, prefix, S/'AD/c2/result.json')
    for i, name in enumerate(NAMES):
        for suffix, factor in PERTURBATIONS:
            label = name+'_'+suffix
            source = S/'FD'/label/'result.json'
            result = read(B/'FD'/label/'result.json')
            if result: fd[label] = result
            q = p.copy(); q[i] += factor*h[i]
            path_checks[label] = path_check(result, execution, read(source), source, q)
    expected_blocks = execution.get('prefix_blocks', []) + execution.get('extension_schedule', [])
    actual_blocks = execution.get('blocks', [])
    checkpoint_schedule = bool(len(actual_blocks) == len(expected_blocks)
        and sum(x['count'] for x in expected_blocks) == execution['steps']
        and sum(x['count'] for x in execution.get('prefix_blocks', [])) == execution['prefix_steps']
        and close(sum(x['count']*x['dt_s'] for x in expected_blocks), execution['end_time_s'])
        and all(a['count'] == e['count'] and a['dt_s'] == e['dt_s'] and Path(a['state_path']).is_file()
                and a.get('source') == ('r57_prefix' if i < len(execution.get('prefix_blocks', [])) else 'r58_extension')
                and (i >= len(execution.get('prefix_blocks', [])) or a['state_path'] == e['state_path'])
                and (i < len(execution.get('prefix_blocks', [])) or a['state_path'] == str(B/'AD/c2/checkpoints'/f'{i-len(execution.get("prefix_blocks", [])):04d}.npz'))
                for i, (a, e) in enumerate(zip(actual_blocks, expected_blocks))))
    work, force, recalculated_jvp = [], [], {}
    for i, name in enumerate(NAMES):
        labels = [name+'_'+suffix for suffix, _ in PERTURBATIONS]
        if not all(label in fd for label in labels): continue
        vals = [fd[label]['terminal'] for label in labels]
        for objective, index in [('work', 0), ('force', 1)]:
            coarse = (vals[0][objective]-vals[1][objective])/(2*h[i])
            fine = (vals[2][objective]-vals[3][objective])/h[i]
            ad = vjp['gradient'][i] if objective == 'work' and vjp else None
            if objective == 'force' and name == 'c2' and jvp: ad = jvp['terminal']['gradient'][1]
            row = dict(coefficient=name, baseline_coefficient=p[i], parameter_scale=scales[i], objective=objective,
                h=h[i], AD=ad, FD_h=coarse, FD_half=fine,
                AD_method='complete loading-work VJP' if objective == 'work' and vjp else ('complete c2 JVP' if ad is not None else 'not computed'),
                units='N mm per dimensionless coefficient' if objective == 'work' else 'N per dimensionless coefficient', **gate(ad, coarse, fine, scales[i]))
            value = jvp['terminal'][objective] if jvp else None
            row['baseline_response'] = value
            row['signed_coefficient_scaled_sensitivity'] = p[i]*fine/value if finite(value) and value != 0 else None
            (work if objective == 'work' else force).append(row)
            if name == 'c2' and jvp:
                recalculated_jvp[objective] = dict(AD=jvp['terminal']['gradient'][index], FD=coarse, FD_half=fine,
                    **gate(jvp['terminal']['gradient'][index], coarse, fine, scales[i]))
    duality, null = None, None
    if vjp and jvp:
        g = vjp['gradient']; other = jvp['terminal']['gradient'][0]
        duality = dict(VJP=g[1], JVP=other, **gate(g[1], other, other, scales[1]))
        terms = [a*b for a, b in zip(g, p)]
        null = dict(contraction_Nmm=sum(terms), relative_to_sum_abs_terms=abs(sum(terms))/max(sum(abs(x) for x in terms), 1e-30),
                    role='diagnostic only; no additional acceptance tolerance',
                    interpretation='Positive common scaling of all three coefficients leaves the zero surface unchanged; two independent morphology degrees of freedom')
    guard = read(B/'checks/source_guard_before.json')
    changes = []
    for name, expected in (guard or {}).items():
        file = Path(name)
        if not file.is_absolute() or not file.is_file(): changes.append(dict(path=name, reason='missing or nonabsolute protected path')); continue
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected: changes.append(dict(path=name, reason='sha256 differs'))
    source_pass = bool(guard) and not changes
    vjp_consistency = bool(vjp and vjp.get('complete') is True and len(vjp.get('gradient', [])) == 3
        and all(finite(x) for x in vjp['gradient']) and vjp.get('objective') == 'full_path_loading_work'
        and vjp.get('includes_original_prefix_reverse') is True and vjp.get('steps') == execution['steps']
        and close(vjp.get('compression'), .1, 1e-10) and jvp
        and close(vjp.get('baseline_work'), jvp['terminal']['work']) and close(vjp.get('baseline_force'), jvp['terminal']['force']))
    stored_jvp_match = bool(stored_jvp and all(stored_check_matches(stored_jvp.get(obj), recalculated_jvp.get(obj)) for obj in ['work', 'force']))
    stored_vjp_match = bool(vjp and len(work) == 3 and len(vjp.get('checks', [])) == 3
        and all(s.get('coefficient') == r['coefficient'] and stored_check_matches(s, r) for s, r in zip(vjp['checks'], work))
        and stored_check_matches(vjp.get('c2_VJP_JVP_check'), duality))
    stored_halving_match = bool(stored_halving and len(stored_halving) == 3 and len(work) == 3
        and all(s.get('coefficient') == r['coefficient'] and stored_check_matches(s, gate(r['FD_half'], r['FD_h'], r['FD_half'], r['parameter_scale']))
                for s, r in zip(stored_halving, work)))
    complete = bool(vjp_consistency and len(fd) == 12 and all(x['passed'] for x in path_checks.values()) and checkpoint_schedule)
    checks = dict(frozen_protocol=frozen_protocol, all_full_paths=all(x['passed'] for x in path_checks.values()),
        complete_checkpoint_schedule=checkpoint_schedule, VJP_scope_and_terminal=vjp_consistency,
        all_three_work_AD_FD=bool(len(work) == 3 and all(r['passed'] is True for r in work)),
        all_three_work_FD_halving=bool(len(work) == 3 and all(r['FD_halving_passed'] for r in work)),
        c2_full_JVP_work=bool(recalculated_jvp.get('work', {}).get('passed')),
        c2_VJP_JVP=bool(duality and duality['passed']),
        stored_JVP_checks_match=stored_jvp_match, stored_VJP_checks_match=stored_vjp_match,
        stored_FD_halving_checks_match=stored_halving_match, protected_files_unchanged=source_pass,
        reused_geometry=bool(geo and geo.get('passed')), reused_r57_acceptance=bool(oldmetrics and oldmetrics.get('final_acceptance_passed')),
        prefix_equivalence=bool(equivalence and equivalence.get('passed') and finite(equivalence.get('response_relative'))
            and finite(equivalence.get('JVP_relative')) and max(equivalence['response_relative'], equivalence['JVP_relative']) < 1e-8),
        extension_short_duality=bool(bench and bench.get('passed') and finite(bench.get('short_duality_relative'))
            and bench['short_duality_relative'] < 1e-7))
    passed = all(checks.values()) if ended else None
    if ended and passed: state = '完整0–10%加载功三系数梯度通过'
    elif ended and complete: state = '完整结果已产生，本轮验收未通过'
    elif ended: state = '执行已停止，验证未完成；保留已有结果'
    else: state = '阶段进行中或待执行，尚无完整验收结论'
    oldvjp = read(S/'AD/work_VJP_result.json')
    costs = dict(prefix_historical_total_s=(oldvjp or {}).get('total_wall_s'),
        prefix_historical_cost_included_in_new_total=False,
        new_JVP_extension_s=(jvp or {}).get('extension_wall_s'),
        new_FD_extension_sum_s=sum(x.get('wall_s', 0.) for x in fd.values()),
        new_VJP_including_reused_prefix_reverse_s=(vjp or {}).get('reverse_wall_s'),
        new_total_wall_s=(vjp or {}).get('total_wall_s'),
        observed_elapsed_since_protocol_s=max(0., time.time()-protocol['started_unix']),
        worker_reported_wall_s=status.get('wall_s'), budget_s=protocol['total_wall_budget_s'],
        new_checkpoint_bytes=sum(x.stat().st_size for x in (B/'AD/c2/checkpoints').glob('*.npz')),
        reused_prefix_checkpoint_bytes=sum(Path(x['state_path']).stat().st_size for x in execution.get('prefix_blocks', []) if Path(x['state_path']).is_file()),
        note='Old r57 143.55min is provenance only. r58 reverse revisits the old prefix and is newly incurred work. Task subtotals omit compilation/preflight/IO and do not replace total wall time.')
    samples = [r for result in ([jvp] if jvp else [])+list(fd.values()) for r in result.get('path', [])]
    metrics = dict(scope='diverse_28 actual equation coefficients; fixed 0-to-10pct full loading-work gradients; terminal force AD c2 only',
        generated_unix=time.time(), conclusion=state, worker_status=worker_status, execution=execution,
        final_acceptance_passed=passed, complete_work_result=complete, acceptance_checks=checks,
        work=work, terminal_force=force, c2_full_JVP_recalculated=recalculated_jvp,
        c2_full_JVP_stored=stored_jvp, c2_VJP_JVP=duality, common_scaling_null_direction=null,
        path_checks=path_checks, completed_FD_result_files=len(fd), protected_file_count=len(guard or {}),
        protected_original_changes=changes, benchmark=bench, prefix_equivalence=equivalence, costs=costs,
        sampled_min_required_J=min((r['min_required_J'] for r in samples if finite(r.get('min_required_J'))), default=None), failure=failure)
    write(A/'metrics.json', metrics)
    columns = ['coefficient', 'baseline_coefficient', 'parameter_scale', 'objective', 'AD_method', 'AD', 'FD_h', 'FD_half', 'h', 'units',
               'near_zero', 'error_type', 'AD_FD_error', 'FD_halving_error', 'applicable_limit', 'FD_halving_passed', 'signed_coefficient_scaled_sensitivity', 'passed']
    with (A/'gradient_table.csv').open('w', newline='', encoding='utf-8-sig') as file:
        out = csv.DictWriter(file, fieldnames=columns, extrasaction='ignore'); out.writeheader(); out.writerows(work+force)
    figures = plots(jvp, fd, work, path_checks, prefix['terminal']['compression'] if prefix else .005)
    def num(x): return '未完成/未记录' if x is None else f'{x:.9g}'
    def minutes(x): return num(x/60 if x is not None else None)
    text = f'''# r58：diverse_28真实方程系数0–10%梯度验证

**{state}。** Worker阶段：{status.get('stage', '未知')}；汇总验收：{passed}。阶段进行中时，缺少后续结果表示待完成，不记为失败。本脚本仅CPU读取和绘图，没有启动FEM、AD、FD或任何自动队列。

## 对象、完整链条与复用范围

曲面方程为f=c1 cosX cosZ+c2 sinX sinY sinZ+c0=0，X=2πx/L（Y、Z同理）。基线(c1,c2,c0)=(-3.4,4.6,2)，三个系数均无量纲。厚度0.5mm来自真实周期三角中面的距离，界面宽度0.05mm固定；函数值不是物理距离，c0不是厚度。三个原始坐标只有两个独立形态自由度：共同正比例缩放不改变零面。

复用受保护r57的0–0.5%实际几何FD终态、c2状态及切向、原始正向检查点和累计功。各FD沿自己的扰动状态继续，累计目标及其导数不清零。新增完整功VJP必须从10%反向经过全部新增块及原0–0.5%块，累积场伴随，再做一次初态和一次参数拉回；并非仅对0.5–10%增量求导。完整设计链条包含隐式几何根、三角面距离、占据、材料、质量、宏观惯性、端缘插值/面积、初始加速度、全部推进与目标。

仍为N32 HEX27/27点/HRZ、共享NH、共同中面端缘等效压板、XY周期和自由横向宏观伸缩；没有新增全壁面或自接触。终点10%，时间2ms，保持原20%/4ms加载曲线；总步数{execution['steps']}，前缀{execution['prefix_steps']}步，新增{execution.get('extension_steps')}步。主步长{num(execution['dt_s'])}s；末块实际步长{num(expected_blocks[-1]['dt_s']) if expected_blocks else '未记录'}s，汇总按各块dt逐段核对，不把末尾补齐步当成同一dt。

## 门槛、覆盖和独立核对

主目标W=∫F dδ（N·mm）：三个系数完整VJP，各核对±h及±h/2真实扰动FD；c2另做完整JVP。h=(3.4e−5,4.6e−5,2e−5)。非近零AD/细FD差与FD减半差均≤1%；|细FD|×参数尺度/目标尺度<1e−6时，改用≤1e−5的缩放绝对误差。参数尺度(3.4,4.6,2)，目标尺度功1N·mm、力1N。近零核对量不是百分比。

末端力只是辅助：c2有完整JVP及双档FD；c1/c0只有FD，没有力VJP。辅助力不通过不会自动改变预先限定的加载功门槛，也不能写作三个力梯度AD全部通过。c2完整功JVP门槛重新计算为{recalculated_jvp.get('work', {}).get('passed')}；完整力JVP为{recalculated_jvp.get('force', {}).get('passed')}。

本汇总逐一复核13条路径的complete、实际参数、原前缀逐行相等、块步数/时间、全部终态10%/2ms、检查点存在和完整前缀反向标志；独立重算VJP/FD、FD减半、JVP/FD及VJP/JVP并比对worker记录。保护文件{len(guard or {})}项，变更/缺失{len(changes)}项。短块对偶仅为先行检查，不代替完整路径。详细检查及每条路径原因见[metrics.json](metrics.json)。

## 数值结果

|系数|完整功VJP|功FD(h)|功FD(h/2)|AD/细FD核对量|FD减半核对量|近零|通过|
|---|---:|---:|---:|---:|---:|---|---|
'''
    for row in work:
        text += '|'+ '|'.join([row['coefficient'], num(row['AD']), num(row['FD_h']), num(row['FD_half']), num(row['AD_FD_error']), num(row['FD_halving_error']), str(row['near_zero']), str(row['passed'])])+'|\n'
    text += '\n功导数单位N·mm/单位无量纲系数。非近零核对量0.01=1%；近零项用缩放绝对量。未计算AD在CSV中留空，不能解读为零。[完整数值与辅助力表](gradient_table.csv)。\n'
    if duality: text += f"\nc2功VJP/JVP核对量{num(duality['AD_FD_error'])}，通过{duality['passed']}。共同缩放方向∇W·p={num(null['contraction_Nmm'])}N·mm，相对各项绝对值之和{num(null['relative_to_sum_abs_terms'])}；此项仅诊断，不事后增加或放宽门槛。\n"
    for row in force: text += f"\n- 辅助末端力{row['coefficient']}：AD={num(row['AD'])}，细FD={num(row['FD_half'])}N/单位系数，AD核对={row['passed']}，FD减半={row['FD_halving_passed']}。\n"
    for image in figures: text += f'\n![{image}]({image})\n'
    text += f'''
## 新增成本与失败记录

r57历史总墙钟{minutes(costs['prefix_historical_total_s'])}min仅作为复用来源，未再次计入r58新增费用。r58新增c2 JVP延续{minutes(costs['new_JVP_extension_s'])}min，已完成FD延续累计{minutes(costs['new_FD_extension_sum_s'])}min，新完整VJP反向{minutes(costs['new_VJP_including_reused_prefix_reverse_s'])}min（包含重新反向旧前缀的实际新费用）。r58最终总墙钟{minutes(costs['new_total_wall_s'])}min；预处理、编译、I/O等可能不在分项内，不能用分项和替代总墙钟。未完成时仅记录观察时距协议开始{minutes(costs['observed_elapsed_since_protocol_s'])}min，不冒充最终计算成本。总预算480min。新检查点{costs['new_checkpoint_bytes']/1024**3:.3f}GiB，复用旧检查点{costs['reused_prefix_checkpoint_bytes']/1024**3:.3f}GiB。

已出现执行失败：{failure.get('message') if failure else '未发现本轮failure.json'}。所有部分路径保留。若worker停止而反向未完成，不能把已有JVP或双档FD当作三系数完整VJP通过；不追加FD阶梯、不放宽门槛、不自动恢复计算。

## 解释边界与证据位置

通过只支持diverse_28该基线附近、此次固定0–10%离散动态路径的加载功设计梯度一致性。有限采样最小必要域J={num(metrics['sampled_min_required_J'])}，不构成全过程正J证明；低占据实际负J不能被此量掩盖。几何导数依赖固定三角连接、同根分支和稳定端缘/最近三角面分区；不能推广到任意系数改变、拓扑改变或20%路径。固定路径AD不含接受/拒绝、减步及步数取整决策导数。

AD/FD通过验证同一离散模型的导数，不证明Abaqus或真实结构的敏感度、屈曲峰位或接触精度。三个有梯度的坐标不等于三个独立形态参数；有符号系数的梯度也不等于其绝对值的梯度。原r53失败及暂停状态保持，heartbeat、Abaqus、训练和Git继续暂停。

正式WSL：{B}；复用来源：{S}。协议/执行计划见../input，完整VJP见../AD/work_VJP_result.json，详细核对见metrics.json与gradient_table.csv。Windows output目录仅同步阅读证据和图片，不复制大数组或检查点。worker原始终态在worker_status_before_analysis.json保留；本汇总不更新项目导航。
'''
    (A/'README.md').write_text(text, encoding='utf-8')
    (B/'README.md').write_text('# r58：真实曲面方程三个系数0–10%\n\n**'+state+'。**\n\n[完整结果与图](analysis/README.md) · [数值表](analysis/gradient_table.csv) · [当前状态](status.json)\n\n复用r57原0–0.5%实际状态，完整VJP须覆盖全部0–10%路径。c2之外末端力仅FD。旧r53门槛、失败与暂停保留；无自动新队列。\n', encoding='utf-8')
    # Never overwrite a live worker's status; retain the original terminal snapshot once.
    latest = read(B/'status.json') or {}
    if ended and latest == worker_status:
        snapshot = A/'worker_status_before_analysis.json'
        if not snapshot.exists(): write(snapshot, worker_status)
        latest.update(analysis='analysis/README.md', analysis_completed_unix=time.time(), final_acceptance_passed=passed,
                      analysis_conclusion=state, worker_status_snapshot='analysis/worker_status_before_analysis.json')
        write(B/'status.json', latest)
    O.mkdir(parents=True, exist_ok=True)
    evidence = ['status.json', 'README.md', 'failure.json', 'input/protocol.json', 'input/execution.json',
                'checks/source_guard_before.json', 'checks/benchmark.json', 'checks/prefix_equivalence.json',
                'checks/full_JVP_c2.json', 'checks/work_FD_halving.json', 'AD/work_VJP_result.json', 'AD/work_VJP_progress.json', 'AD/c2/result.json']
    evidence += ['FD/'+label+'/result.json' for label in fd]
    for rel in evidence:
        source = B/rel
        if source.is_file():
            destination = O/rel; destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, destination)
    for source in A.iterdir():
        if source.is_file() and source.suffix in {'.json', '.csv', '.png', '.md'}:
            destination = O/'analysis'/source.name; destination.parent.mkdir(exist_ok=True); shutil.copy2(source, destination)
    print(json.dumps(dict(worker_stage=worker_status.get('stage'), complete=complete, final_acceptance_passed=passed,
                          FD_result_files=len(fd), report=str(O/'analysis/README.md')), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int, help='Wait once for this known worker to exit, bounded by protocol deadline +900s')
    main(parser.parse_args().wait_pid)
