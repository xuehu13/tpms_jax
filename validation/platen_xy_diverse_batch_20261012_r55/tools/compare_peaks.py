"""Same predeclared r54 peak protocol, with case-aware report labels."""
from pathlib import Path
import hashlib,json,sys
B=Path(__file__).resolve().parents[1];R=B.parents[1];kind=sys.argv[1];E=B/kind;cfg=json.loads((E/'input/protocol.json').read_text())
source=R/'validation/platen_xy_diverse04_20261012_r54/tools/compare_peaks.py';original=source.read_text();code=original.replace('diverse_04',kind).replace('r54','r55')
old='复用原'+kind+'的8372节点／15914三角中面和原N32积分点占据场，不重新拟合几何。'
new=f"复用原{kind}的{cfg['physical_nodes']}节点／{cfg['shell_elements']}三角中面，在自身实际N32积分点生成占据场，不拟合几何或体积分数。"
assert old in code;code=code.replace(old,new)
old='263条独立节点关系形成1578条壳方程；上下每端162个中面节点接触压板。'
new=f"{cfg['XY_relations']}条独立节点关系形成{cfg['XY_equations']}条壳方程；上下端分别{cfg['rim_node_count']}个中面节点接触压板。"
assert old in code;code=code.replace(old,new)
code=code.replace('platen_xy_diverse04_20261012_r55','platen_xy_diverse_batch_20261012_r55')
code=code.replace('r55_diverse04_midsurface_end_20pct_20261012',f'r55_{kind}_midsurface_end_20pct_20261012')
code=code.replace('原XYZ结果保留，但旧时长、横向约束及停止范围可能不同，不能把新旧差异全部定量归因于取消Z周期或接触简化。','历史曲线仅用于选样，材料、时长、质量与接触条件可能不同，不能当作本轮成对参考或把新旧差定量归因于单一因素。')
(E/'checks/peak_recipe.json').write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(original.encode()).hexdigest(),'adapted_sha256':hashlib.sha256(code.encode()).hexdigest(),'change':'case paths and counts only; peak rules and metrics unchanged'},indent=2)+'\n')
exec(compile(code,str(source),'exec'),{'__name__':'__main__','__file__':str(Path(__file__).resolve())})
