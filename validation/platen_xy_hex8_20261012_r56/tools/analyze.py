"""Read saved HEX8/HEX27/shell paths; no FEM, fitting, or reference writes.

Installed entry: validation/platen_xy_hex8_20261012_r56/tools/analyze.py.
Repeatable while a runner is active: result.json, then path_latest.json fallback.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = Path(__file__).resolve().parents[1]
R = B.parents[1]
COLORS = {"HEX8": "#bd4b22", "HEX27": "#b3892f", "Abaqus": "#245d8c"}
SOURCE_HASHES = {}
TOL_TIME = 1e-9  # Abaqus single-precision final time is slightly below 4.2 ms.


def read(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    raw = path.read_bytes()
    SOURCE_HASHES[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(clean(rows))


def load_result(directory, allow_latest=False):
    directory = Path(directory)
    source = directory / "result.json"
    result = read(source)
    if result is None and allow_latest:
        source = directory / "path_latest.json"
        latest = read(source)
        if latest is not None:
            result = latest if isinstance(latest, dict) else {"path": latest}
            result = {**result, "complete": False, "status": "saved_valid_prefix_running_or_stopped"}
            progress = read(directory / "progress.json", {})
            for key in ("steps", "wall_seconds", "dt_s", "remaining_budget_seconds"):
                if key in progress:
                    result[key] = progress[key]
    if not result or not result.get("path"):
        return None
    original_row_count = len(result["path"])
    valid = []
    for row in result["path"]:
        if all(k in row and np.isfinite(row[k]) for k in ("time", "compression", "force_N")):
            valid.append(row)
    if not valid:
        return None
    # Keep the last sample at repeated times; never extrapolate missing history.
    indexed = {float(row["time"]): row for row in valid}
    result = {**result, "path": [indexed[t] for t in sorted(indexed)], "source": str(source)}
    result["filtered_nonfinite_or_missing_rows"] = len(valid) != original_row_count
    return result


def series(result, key):
    return np.asarray([row.get(key, np.nan) for row in result["path"]], dtype=float)


def loading(result, end_time):
    time = series(result, "time")
    end = min(float(end_time), float(time[-1]))
    mask = time < end - 1e-13
    x = np.r_[series(result, "compression")[mask], np.interp(end, time, series(result, "compression"))]
    f = np.r_[series(result, "force_N")[mask], np.interp(end, time, series(result, "force_N"))]
    t = np.r_[time[mask], end]
    # Original r54 convention: preserve last force at duplicate compression.
    unique, reverse = np.unique(x[::-1], return_index=True)
    return unique, f[len(x) - 1 - reverse], (x, f, t)


def grid_for(emax):
    return np.linspace(0.0, emax, max(2, round(emax / 0.0002) + 1))


def mean_window(result, key, lo, hi):
    time = series(result, "time")
    vals = series(result, key)
    if time[0] > lo + TOL_TIME or time[-1] < hi - TOL_TIME or not np.isfinite(vals).all():
        return None
    # Tolerance only admits floating-point endpoint representation, not absent data.
    actual_hi = min(hi, time[-1])
    t = np.r_[lo, time[(time > lo) & (time < actual_hi)], actual_hi]
    return float(np.trapezoid(np.interp(t, time, vals), t) / (actual_hi - lo))


def ratio(a, b):
    return None if a is None or b is None or abs(b) < 1e-30 else float(a / b)


def peak_info(result, end_time):
    x, f, raw = loading(result, end_time)
    eps, force, times = raw
    idx = int(np.argmax(force))
    endpoint = bool(eps[idx] >= x[-1] - 1e-8)
    grid = grid_for(float(x[-1]))
    smooth = uniform_filter1d(np.interp(grid, x, f), size=25, mode="nearest")
    threshold = 0.1 * float(np.max(np.abs(smooth)))
    inds, props = find_peaks(smooth, prominence=threshold, distance=25)
    gaps = np.diff(series(result, "time"))
    return {
        "scope_terminal_compression": float(x[-1]),
        "raw_global_max": {"compression_percent": float(eps[idx] * 100), "force_N": float(force[idx]),
                           "time_s": float(times[idx]), "at_loading_endpoint": endpoint,
                           "is_interior_global_max": bool(idx > 0 and not endpoint)},
        "raw_sample_time_spacing_s_min_max": [float(gaps.min()), float(gaps.max())] if gaps.size else None,
        "smoothing_points": 25, "smoothing_width_percentage_points": float(25 * (grid[1] - grid[0]) * 100),
        "prominence_threshold_N": threshold,
        "prominent_peaks": [{"compression_percent": float(grid[i] * 100), "force_N": float(smooth[i]),
                              "prominence_N": float(p)} for i, p in zip(inds, props["prominences"])],
        "no_peak_extrapolation": True,
    }


def pair_metrics(left, right, shell, load_time, L, hold_window):
    common_time = min(load_time, series(left, "time")[-1], series(right, "time")[-1])
    curves = [loading(v, common_time) for v in (left, right, shell)]
    emax = min(float(c[0][-1]) for c in curves)
    grid = grid_for(emax)
    a, b = [np.interp(grid, c[0], c[1]) for c in curves[:2]]
    # Frozen r54 rule: shell's own raw peak within the actual common window.
    den = float(np.max(np.abs(curves[2][2][1])))
    work = [float(np.trapezoid(c[1], c[0] * L)) for c in curves]
    full_loading = common_time >= load_time - TOL_TIME and emax >= 0.2 - 1e-8
    paired_result_complete = bool(left.get("complete", False) and right.get("complete", False)
                                  and not left.get("filtered_nonfinite_or_missing_rows", False)
                                  and not right.get("filtered_nonfinite_or_missing_rows", False))
    left_hold = mean_window(left, "force_N", *hold_window)
    right_hold = mean_window(right, "force_N", *hold_window)
    hold_bias = ratio(left_hold, right_hold)
    hold_bias = None if hold_bias is None else hold_bias - 1
    peaks = [peak_info(v, common_time) for v in (left, right)]
    first = [p["prominent_peaks"][0] if p["prominent_peaks"] else None for p in peaks]
    secant = [float(np.interp(.02, c[0], c[1])) / (.02 * L) if emax >= .02 else None for c in curves[:2]]
    rms = ratio(float(np.sqrt(np.mean((a - b) ** 2))), den)
    work_bias = ratio(work[0], work[1])
    return {
        "common_time_s": float(common_time), "common_compression_max": emax,
        "full20_loading": bool(full_loading), "full_hold_window": left_hold is not None and right_hold is not None,
        "paired_result_complete": paired_result_complete,
        "normalizer_N": den, "normalizer_scope": "shell raw loading peak in this pair's common achieved window",
        "curve_RMS_over_shell_peak": rms,
        "loading_work_left_N_mm": work[0], "loading_work_right_N_mm": work[1], "loading_work_shell_N_mm": work[2],
        "loading_work_relative_bias_vs_right": None if work_bias is None else work_bias - 1,
        "loading_work_difference_over_shell_work": ratio(work[0] - work[1], work[2]),
        "hold_mean_left_N": left_hold, "hold_mean_right_N": right_hold, "hold_mean_relative_bias_vs_right": hold_bias,
        "secant_stiffness_2pct_left_N_per_mm": secant[0], "secant_stiffness_2pct_right_N_per_mm": secant[1],
        "secant_stiffness_2pct_ratio_left_over_right": ratio(*secant),
        "raw_global_max": {"left": peaks[0]["raw_global_max"], "right": peaks[1]["raw_global_max"]},
        "raw_global_max_location_difference_percentage_points": peaks[0]["raw_global_max"]["compression_percent"] - peaks[1]["raw_global_max"]["compression_percent"],
        "first_prominent_peak_difference_percentage_points": None if None in first else first[0]["compression_percent"] - first[1]["compression_percent"],
        "prominent_peaks": {"left": peaks[0], "right": peaks[1]},
        "three_targets_within10pct": bool(rms <= .1 and abs(work_bias - 1) <= .1 and abs(hold_bias) <= .1)
        if paired_result_complete and full_loading and hold_bias is not None and rms is not None and work_bias is not None else None,
    }


def solver_health(result, L, hold_window):
    time = series(result, "time")
    ie_key = "energy_N_mm" if "energy_N_mm" in result["path"][0] else "ALLIE"
    ke_key = "KE_N_mm" if "KE_N_mm" in result["path"][0] else "ALLKE"
    info = {k: v for k, v in result.items() if k != "path"}
    for key in ("required_positive_J_min", "all_J_min"):
        vals = series(result, key)
        info["sampled_min_" + key] = float(np.nanmin(vals)) if np.isfinite(vals).any() else None
    info["sampled_max_negative_void_J_points"] = max((r.get("negative_void_J_points", 0) for r in result["path"]), default=0)
    info["terminal_time_s"] = float(time[-1])
    info["terminal_compression"] = float(series(result, "compression")[-1])
    info["terminal_KE_over_IE"] = ratio(series(result, ke_key)[-1], series(result, ie_key)[-1])
    info["hold_upper_mean_N"] = mean_window(result, "force_N", *hold_window)
    info["hold_lower_mean_N"] = mean_window(result, "bottom_force_N", *hold_window)
    upper, lower = info["hold_upper_mean_N"], info["hold_lower_mean_N"]
    info["hold_upper_minus_lower_N"] = None if upper is None or lower is None else upper - lower
    if ie_key == "energy_N_mm":
        x = series(result, "compression") * L
        f = series(result, "force_N")
        work = np.r_[0., np.cumsum(.5 * (f[1:] + f[:-1]) * np.diff(x))]
        total = series(result, ie_key) + series(result, ke_key) + series(result, "contact_energy_N_mm")
        info["sampled_energy_work_max_gap_over_max_abs_work"] = ratio(float(np.nanmax(abs(total - total[0] - work))), float(np.max(abs(work))))
    info["actual_J_scope"] = "saved accepted samples only; no every-step proof; actual F/J unmodified"
    info["hold_interpretation"] = "short dynamic hold; upper/lower force imbalance is not a static stiffness error"
    return info


def fmt(value, percent=False, digits=3):
    return "—" if value is None else f"{100 * value if percent else value:.{digits}f}"


def plot_case(kind, solvers, out, load_time):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for name, result in solvers.items():
        c = COLORS[name]
        x, f, _ = loading(result, load_time)
        for ax in axes[0]:
            ax.plot(x * 100, f, color=c, lw=1.35, label=name)
        t = series(result, "time") * 1000
        axes[1, 0].plot(t, series(result, "force_N"), color=c, lw=1, label=name + " upper")
        axes[1, 0].plot(t, series(result, "bottom_force_N"), color=c, ls="--", lw=.85, label=name + " lower")
        ie_key = "energy_N_mm" if "energy_N_mm" in result["path"][0] else "ALLIE"
        ke_key = "KE_N_mm" if "KE_N_mm" in result["path"][0] else "ALLKE"
        axes[1, 1].plot(t, series(result, ie_key), color=c, label=name + " internal")
        axes[1, 1].plot(t, series(result, ke_key), color=c, ls="--", label=name + " kinetic")
    axes[0, 0].set(xlim=(0, 20), xlabel="Engineering compression (%)", ylabel="Upper-platen force (N)", title="Actual achieved loading histories")
    axes[0, 1].set(xlim=(0, 6), xlabel="Engineering compression (%)", ylabel="Upper-platen force (N)", title="Initial window; no force scaling")
    axes[1, 0].set(xlabel="Physical time (ms)", ylabel="Compressive force (N)", title="Upper and lower platen forces")
    axes[1, 1].set(xlabel="Physical time (ms)", ylabel="Energy (N mm)", title="Internal and kinetic energy (different discretizations)")
    for ax in axes.ravel():
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(kind + " | HEX8 / original HEX27 / unchanged shell")
    fig.savefig(out / "three_solver_comparison.png", dpi=190)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    for name, result in solvers.items():
        t = series(result, "time") * 1000
        c = COLORS[name]
        j = series(result, "required_positive_J_min")
        if np.isfinite(j).any():
            axes[0].plot(t, j, color=c, label=name + " required positive J")
            axes[0].plot(t, series(result, "all_J_min"), color=c, ls="--", label=name + " all J")
        ie = series(result, "energy_N_mm" if "energy_N_mm" in result["path"][0] else "ALLIE")
        ke = series(result, "KE_N_mm" if "KE_N_mm" in result["path"][0] else "ALLKE")
        axes[1].plot(t, np.divide(ke, ie, out=np.full_like(ke, np.nan), where=abs(ie) > 1e-20), color=c, label=name)
        axes[2].plot(t, series(result, "force_N") - series(result, "bottom_force_N"), color=c, label=name)
    axes[0].set(ylabel="Actual det(F), saved samples", title="No determinant clipping; shell has no comparable Gauss-field J")
    axes[1].set(ylabel="Kinetic / internal energy", ylim=(0, None))
    axes[2].set(xlabel="Physical time (ms)", ylabel="Upper minus lower force (N)", title="Dynamic force imbalance; not static stiffness")
    for ax in axes:
        ax.grid(alpha=.2)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
    fig.savefig(out / "health_and_dynamics.png", dpi=190)
    plt.close(fig)


def evidence_files(case, ref):
    result = {}
    for label, path in {
        "HEX8_preparation": case / "checks/preparation.json",
        "HEX8_input": case / "jax/input.json",
        "HEX8_costs": case / "costs.json",
        "HEX8_execution_receipt": case / "checks/execution_receipt.json",
        "HEX27_input": ref / "jax/input.json",
        "HEX27_parameters": ref / "input/protocol.json",
        "HEX27_preparation": ref / "checks/gauss_preparation.json",
        "original_costs": ref / "costs.json",
    }.items():
        value = read(path)
        if value is not None:
            result[label] = {"path": str(path), "data": value}
    return result


def analyze_case(kind, protocol):
    case = B / kind
    ref = Path(protocol["references"][kind]["case_dir"])
    result8 = load_result(case / "jax", allow_latest=True)
    if result8 is None:
        return None
    solvers = {"HEX8": result8, "HEX27": load_result(ref / "jax"), "Abaqus": load_result(ref / "abaqus")}
    if any(value is None for value in solvers.values()):
        raise FileNotFoundError(f"{kind}: original HEX27 or Abaqus result.json missing; no reference substitution")
    out = case / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    load_time = float(protocol.get("load_time_s", .004))
    L = float(protocol.get("L_mm", 10.))
    hold_window = tuple(protocol.get("metrics", {}).get("hold_mean_window_s", [.0041, .0042]))
    evidence = evidence_files(case, ref)
    pairs = {}
    for left, right in [("HEX8", "Abaqus"), ("HEX27", "Abaqus"), ("HEX8", "HEX27")]:
        pairs[left + "_vs_" + right] = pair_metrics(solvers[left], solvers[right], solvers["Abaqus"], load_time, L, hold_window)
        pairs[left + "_vs_" + right]["three_target_evaluation_applies"] = right == "Abaqus"
        if right != "Abaqus":
            pairs[left + "_vs_" + right]["three_targets_within10pct"] = None
    common_time = min(load_time, *(series(v, "time")[-1] for v in solvers.values()))
    triple = {name: loading(value, common_time) for name, value in solvers.items()}
    grid = grid_for(min(float(v[0][-1]) for v in triple.values()))
    rows = []
    interpolated = {name: np.interp(grid, value[0], value[1]) for name, value in triple.items()}
    smoothed = {name: uniform_filter1d(values, size=25, mode="nearest") for name, values in interpolated.items()}
    for i, eps in enumerate(grid):
        row = {"compression_percent": eps * 100}
        for name in solvers:
            row[name + "_force_N"] = interpolated[name][i]
            row[name + "_fixed_moving_mean_N"] = smoothed[name][i]
        rows.append(row)
    write_csv(out / "three_curve_common_window.csv", rows)
    actual = []
    for name, result in solvers.items():
        for row in result["path"]:
            actual.append({"solver": name, **row})
    fields = list(dict.fromkeys(k for row in actual for k in row))
    write_csv(out / "all_actual_paths.csv", actual, fields)
    health = {name: solver_health(value, L, hold_window) for name, value in solvers.items()}
    peaks = {name: peak_info(value, load_time) for name, value in solvers.items()}
    peak_rows = []
    for name, value in peaks.items():
        peak_rows.append({"solver": name, "definition": "raw global max in own achieved loading", **value["raw_global_max"]})
        peak_rows.extend({"solver": name, "definition": "fixed moving mean prominent interior peak", **v} for v in value["prominent_peaks"])
    write_csv(out / "peak_summary.csv", peak_rows, ["solver", "definition", "compression_percent", "force_N", "time_s", "at_loading_endpoint", "prominence_N"])
    prep_data = evidence.get("HEX8_preparation", {}).get("data", {})
    prep_summary = {key: prep_data.get(key) for key in (
        "occupancy_volume_fraction", "HEX27_occupancy_volume_fraction", "occupancy_relative_change",
        "total_mass_tonne", "contact_area_mm2", "preparation_wall_seconds", "pass", "node_count",
        "Gauss_points_per_cell", "mass_lumping")}
    metrics = {
        "kind": kind, "original_case_dir": str(ref), "HEX8_result_complete": bool(result8.get("complete", False)),
        "triple_common_compression_max": float(grid[-1]), "triple_common_time_s": float(common_time),
        "shell_full_loading_raw_peak_N": float(np.max(np.abs(loading(solvers["Abaqus"], load_time)[2][1]))),
        "pairs": pairs, "own_achieved_window_peaks": peaks, "health": health, "evidence": evidence,
        "preparation_summary": prep_summary,
        "no_matching_fitted_params": True, "fitting_operations_performed_by_analysis": False,
        "new_FEM_AD_FD_calls": False,
        "peak_rule": "0.0002 compression grid, centered nearest-edge moving mean25; prominence>=10% own smoothed maximum absolute force, distance>=25; all peaks listed",
        "partial_rule": "Each pair uses its own actual common achieved window; old HEX27 diverse_05 is never extended. Full HEX8-shell evidence, when available, is separate from triple common-window evidence.",
        "interpretation": "Element-family/discretization comparison under common end-rim simplification; neither shell truth nor contact/AD certification. No curve scaling, shifting, time warping, or parameter fitting.",
        "source_sha256": dict(SOURCE_HASHES),
    }
    write_json(out / "metrics.json", metrics)
    pair_rows = [{"comparison": name, **{k: v for k, v in value.items() if not isinstance(v, dict)}} for name, value in pairs.items()]
    write_csv(out / "pair_metrics.csv", pair_rows)
    plot_case(kind, solvers, out, load_time)
    report = [f"# {kind}：HEX8／原HEX27／原Abaqus壳只读对照", "",
              "本报告只读取保存路径，没有FEM、AD/FD、参数匹配、力缩放、曲线平移或时间扭曲。原壳及原HEX27数据不改写。",
              f"原参考目录：`{ref}`。新HEX8状态：`{result8.get('status', 'unknown')}`；结果complete={bool(result8.get('complete', False))}。",
              f"三曲线共同加载范围0–{grid[-1] * 100:.4f}%；HEX8–壳与HEX27–壳各自共同窗口另列，不能把旧05的缺失20%／保载补成已完成。", "",
              "| 比较（左减右） | 共同末压缩% | RMS／壳峰% | 加载功相对右侧差% | 4.1–4.2ms上板均力差% | 2%割线刚度比 | 首显著峰差/百分点 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, value in pairs.items():
        report.append(f"| {name} | {fmt(value['common_compression_max'], True)} | {fmt(value['curve_RMS_over_shell_peak'], True)} | {fmt(value['loading_work_relative_bias_vs_right'], True)} | {fmt(value['hold_mean_relative_bias_vs_right'], True)} | {fmt(value['secant_stiffness_2pct_ratio_left_over_right'])} | {fmt(value['first_prominent_peak_difference_percentage_points'])} |")
    report += ["", "RMS分母继承r54：本比较实际共同窗口内壳原始上板峰；原壳自身完整加载峰另存在metrics.json。功为上板力对上板行程积分。HEX8–HEX27的功相对差分母为HEX27，同时另存除以壳功的差值。2%割线刚度定义F(2%)/(0.02L)，仅为初始段诊断量。", "",
               "峰位同时保留原始全局最大值和固定平滑显著内部峰；端点最大值不称屈曲峰，没有合格峰则不强配。平滑只用于诊断，原始力／功／RMS不平滑。", "",
               "![三求解器力、上下板力和能量](three_solver_comparison.png)", "![实际J与动态指标](health_and_dynamics.png)", "",
               "| 求解器 | 保存末时间ms | 保存末压缩% | 采样主体最小实际J | 采样全域最小实际J | 末KE/IE% | 保载上板均力N | 保载下板均力N | 上下差N | 计算墙钟min |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, value in health.items():
        report.append(f"| {name} | {fmt(value['terminal_time_s'] * 1000)} | {fmt(value['terminal_compression'], True)} | {fmt(value['sampled_min_required_positive_J_min'])} | {fmt(value['sampled_min_all_J_min'])} | {fmt(value['terminal_KE_over_IE'], True)} | {fmt(value['hold_upper_mean_N'])} | {fmt(value['hold_lower_mean_N'])} | {fmt(value['hold_upper_minus_lower_N'])} | {fmt(value.get('wall_seconds', 0) / 60) if 'wall_seconds' in value else '—'} |")
    report += ["", "保载是原4.1–4.2ms短动态窗口，不能把上板力差叫静态刚度差；上下力差、动能与保存态实际J须一起读。实际J仅检查已接受保存样本，低占据区负J仍保留；不是全过程安全证明。", "",
               "| 准备指标 | 原值（未拟合） |", "|---|---:|",
               f"| HEX8占据体积分数 | {fmt(prep_summary['occupancy_volume_fraction'], digits=8)} |",
               f"| HEX27占据体积分数 | {fmt(prep_summary['HEX27_occupancy_volume_fraction'], digits=8)} |",
               f"| 占据体积相对变化% | {fmt(prep_summary['occupancy_relative_change'], True)} |",
               f"| HEX8总质量tonne | {prep_summary['total_mass_tonne'] if prep_summary['total_mass_tonne'] is not None else '—'} |",
               f"| 上下端参考接触面积mm² | {prep_summary['contact_area_mm2'] if prep_summary['contact_area_mm2'] is not None else '—'} |",
               f"| 每单元积分点数 | {prep_summary['Gauss_points_per_cell'] if prep_summary['Gauss_points_per_cell'] is not None else '—'} |",
               f"| 准备墙钟s | {fmt(prep_summary['preparation_wall_seconds'])} |", "",
               "体积分数、质量（HEX8行和／HEX27 HRZ）、接触面积、输入哈希和准备／计算成本按checks/preparation.json、jax/input.json等原字段完整收入metrics.json的evidence；不调整厚度、占据或质量使其匹配参考。HEX8积分点、位移插值及质量分布随离散改变，不应说成完全同一离散模型。", "",
               "本轮只回答已声明的共同中面端缘／XY周期模型下离散变化的影响，不认证完整有限厚度壁面接触、自接触、任意构型或设计梯度。全部三项约10%工作目标判断只适用于完整HEX8–壳或HEX27–壳对照；HEX8–HEX27只作离散差异诊断。", "",
               "原路径、共同窗口CSV、峰表、指标、哈希和健康量均保存在本目录。"]
    (out / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return metrics, solvers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs="+", help="Optional case filter; no solver is invoked")
    args = parser.parse_args()
    protocol = read(B / "protocol.json")
    if protocol is None:
        raise FileNotFoundError(B / "protocol.json")
    names = list(protocol["cases"])
    if args.case:
        unknown = set(args.case) - set(names)
        if unknown:
            raise ValueError(f"Unknown cases {sorted(unknown)}")
        names = [name for name in names if name in args.case]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    done, pending, errors = {}, [], {}
    for kind in names:
        try:
            value = analyze_case(kind, protocol)
            if value is None:
                pending.append(kind)
            else:
                done[kind] = value
        except Exception as exc:
            errors[kind] = {"type": type(exc).__name__, "message": str(exc)}
    out = B / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for kind, (metrics, _) in done.items():
        for comparison, values in metrics["pairs"].items():
            summaries.append({"kind": kind, "comparison": comparison,
                              **{key: value for key, value in values.items() if not isinstance(value, dict)}})
    write_csv(out / "summary.csv", summaries)
    write_json(out / "summary.json", {"cases": {k: v[0] for k, v in done.items()}, "pending_no_valid_path": pending,
                                     "analysis_errors": errors, "no_matching_fitted_params": True,
                                     "new_FEM_AD_FD_calls": False, "source_sha256": dict(SOURCE_HASHES)})
    if done:
        fig, axes = plt.subplots(len(done), 2, figsize=(12, 3.5 * len(done)), squeeze=False, constrained_layout=True)
        for row, (kind, (_, solvers)) in enumerate(done.items()):
            for name, result in solvers.items():
                x, f, _ = loading(result, float(protocol.get("load_time_s", .004)))
                for ax in axes[row]:
                    ax.plot(x * 100, f, color=COLORS[name], label=name, lw=1.2)
            for col, ax in enumerate(axes[row]):
                ax.set(xlim=(0, 20 if col == 0 else 6), xlabel="Engineering compression (%)", ylabel="Upper force (N)", title=kind + (" | achieved loading" if col == 0 else " | initial window"))
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
        fig.savefig(out / "all_cases_force.png", dpi=190)
        plt.close(fig)
    text = ["# HEX8全积分：原HEX27与原壳对照汇总", "",
            "只读保存路径的重复分析；没有新求解、梯度、参数匹配或原参考改写。各组保留自身实际共同窗口、壳峰分母与4.1–4.2ms保载窗。", "",
            "| 构型 | HEX8–壳共同末压缩% | RMS／壳峰% | 功差% | 保载上板均力差% | 2%割线刚度比 | 完整三目标 |", "|---|---:|---:|---:|---:|---:|---|"]
    for kind, (metrics, _) in done.items():
        values = metrics["pairs"]["HEX8_vs_Abaqus"]
        flag = values["three_targets_within10pct"]
        judgment = "未完整，暂不验收" if flag is None else ("达到本轮限定目标" if flag else "三个目标未全达到")
        text.append(f"| [{kind}](../{kind}/analysis/README.md) | {fmt(values['common_compression_max'], True)} | {fmt(values['curve_RMS_over_shell_peak'], True)} | {fmt(values['loading_work_relative_bias_vs_right'], True)} | {fmt(values['hold_mean_relative_bias_vs_right'], True)} | {fmt(values['secant_stiffness_2pct_ratio_left_over_right'])} | {judgment} |")
    if done:
        text += ["", "![全部已保存三曲线](all_cases_force.png)"]
    text += ["", "旧diverse_05的HEX27预算停点仍为原有效前缀，不因HEX8完成而被补成完整。HEX8–壳完整范围与三曲线共同范围分别保存。RMS不是逐点误差上限，首次显著峰接近不代表峰后分支一致，短保载力差不称静态刚度差。",
             "", "体积分数、质量、实际J及成本见各组metrics.json与准备凭据。约10%是原限定工作目标，壳不是精确真值。"]
    if pending:
        text += ["", "尚无可读HEX8有效路径，跳过：" + "、".join(pending) + "。"]
    if errors:
        text += ["", "分析读取失败（不启动重算）："] + [f"- {k}: `{v['type']}: {v['message']}`" for k, v in errors.items()]
    (out / "README.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps({"analyzed": list(done), "pending": pending, "errors": errors, "new_FEM_calls": False}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
