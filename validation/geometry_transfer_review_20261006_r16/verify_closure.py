from pathlib import Path
import json,hashlib,shutil,subprocess,re
R=Path("/home/xuehu/projects/tpms_jax")
W=Path("/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an")
D=R/"validation/geometry_transfer_review_20261006_r16"
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):h.update(chunk)
    return h.hexdigest()
def write(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
# Distinguish the mutable evidence index from frozen scientific originals.
before=json.loads((D/"frozen_before.json").read_text())["files"]
changed=[s for s,h in before.items() if sha(R/s)!=h]
assert changed==["validation/README.md"],changed
v=json.loads((D/"verification.json").read_text())
v["postprocessing_before_doc_updates_frozen_file_count"]=v.pop("frozen_file_count")
v["all_preexisting_files_unchanged_during_postprocessing"]=v.pop("all_frozen_bytes_unchanged")
v.update(frozen_scientific_file_count=len(before)-1,all_frozen_scientific_bytes_unchanged=True,
    mutable_evidence_index_updated=["validation/README.md"])
# Rejected-field interpolation is diagnostic, not a recovered solver checkpoint.
p=D/"rejected_endpoint_diagnosis.json";obj=json.loads(p.read_text())
obj["reference_interpolation_scope"]="Same regular HEX27 basis, point order, and cached periodic class map. Not a bitwise replay of original FEM reference gradients or time integration; overflow counts describe this readonly reconstruction."
obj["J_determinant_evaluation"]="Numpy determinant for diagnostic volume counts; shared JAX energy/stress kernel uses its existing determinant formula."
write(p,obj)
for p in (D/"REVIEW.md",W/"GEOMETRY_TRANSFER_REVIEW.md"):
    t=p.read_text().replace("第二块用同一真实27点规则、缓存周期类编号、同一材料能量/应力核作只读求值。",
        "第二块用同一真实27点规则、缓存周期类编号、规则背景单元形函数和同一材料能量/应力核作只读求值。积分点与权重对原缓存核对通过，但不是原FEM参考梯度的逐位重放；下列计数描述这个拒绝场的后处理重建，不是原求解器的在线首错记录。")
    p.write_text(t)
for p in (W/"TPMS_RESEARCH_REVIEW.md",R/"docs/TPMS_RESEARCH_REVIEW.md"):
    t=p.read_text().replace("依据基线7e9c642及r15输入/诊断","依据基线450bc8f及r15执行/r16限定诊断")
    p.write_text(t)
for p in (W/"PAPER_ROUTE.md",R/"docs/PAPER_ROUTE.md"):
    t=p.read_text().replace("Windows `work/research_synthesis_20261006/literature_review.json`",
        "Windows `history/completed_tools_20261006/research_synthesis_20261006/literature_review.json`")
    p.write_text(t)
# Validate new links, scopes, and all data have strict JSON encodings.
for p in D.rglob("*.json"):json.loads(p.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
diag=json.loads((D/"rejected_endpoint_diagnosis.json").read_text())
assert [r["accepted"] for r in diag["rows"]]==[False]*3
assert all(r["full_domain_energy_N_mm"] is None for r in diag["rows"])
assert [r["complete_finite_nodal_cells"] for r in diag["rows"]]==[0,30846,0]
partial=json.loads((D/"partial_force_diagnosis.json").read_text())
assert len(partial["rows"])==14 and all(partial[k] is None for k in
    ("full_1000_point_RMS","full_20pct_work_difference","20pct_hold_force_difference","mode_difference"))
scope=json.loads((D/"scope_decision.json").read_text())
assert not scope["step2_20pct_credible_transfer_goal_passed"] and not scope["diverse04_t0p50"]["migration_certified"]
assert scope["existing_diverse28_scope"]==json.loads((R/"validation/forward_scope_20261006_r14/forward_scope.json").read_text())
for p in (D/"REVIEW.md",R/"docs/RESEARCH_PLAN.md",R/"docs/RESEARCH_STATUS.md",R/"README.md",R/"AGENTS.md"):
    for target in re.findall(r"\]\(([^)]+)\)",p.read_text()):
        if target.startswith(("http","/")):continue
        assert (p.parent/target.split("#")[0]).exists(),(str(p),target)
assert not re.search("待执行|下一项仅第3步", (W/"PROJECT_OVERVIEW.md").read_text().replace("无自动待执行作业",""))
allowed={"AGENTS.md","README.md","docs/FILE_MAP.md","docs/PAPER_ROUTE.md","docs/RESEARCH_BACKGROUND.md",
"docs/RESEARCH_PLAN.md","docs/RESEARCH_STATUS.md","docs/TPMS_RESEARCH_REVIEW.md","scripts/README.md","validation/README.md"}
assert set(subprocess.check_output(["git","diff","--name-only"],cwd=R,text=True).splitlines())==allowed
assert not subprocess.run(["git","diff","--check"],cwd=R,capture_output=True).returncode
shutil.copyfile(W/"work/geometry_closure_20261006/review.log",D/"readonly_postprocess.log")
shutil.copyfile(__file__,D/"verify_closure.py")
v.update(current_document_links_and_status_checked=True,strict_json_and_scope_checks_passed=True,
    allowed_current_docs_only_modified=True,diff_whitespace_check_passed=True,
    existing_diverse28_scope_exactly_preserved=True,rejected_or_missing_metrics_not_certified=True)
write(D/"verification.json",v)
manifest=json.loads((D/"evidence_manifest.json").read_text())
manifest["mutable_documentation_index_exception"]={"path":"validation/README.md","before_sha256":before["validation/README.md"],"after_sha256":sha(R/"validation/README.md"),"reason":"Current readable index updated; not a frozen scientific result."}
manifest["output_files"]={str(p.relative_to(D)):sha(p) for p in D.rglob("*") if p.is_file() and p.name!="evidence_manifest.json"}
manifest["current_documents"]={s:sha(R/s) for s in sorted(allowed)}
write(D/"evidence_manifest.json",manifest)
print(json.dumps(v,ensure_ascii=False))
