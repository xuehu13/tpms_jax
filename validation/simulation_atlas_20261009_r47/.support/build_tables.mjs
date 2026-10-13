import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

try {
const out=path.resolve(import.meta.dirname,'..');
const read=async name=>JSON.parse(await fs.readFile(path.join(out,name+'.json'),'utf8'));
const [runs,errors,points,statics,coverage,paths]=await Promise.all(['curve_catalog','window_errors','point_errors','static_points','coverage_notes','path_errors'].map(read));
const wb=Workbook.create();
const sheets={};
function tab(name,title,heads,rows,widths){
  const s=wb.worksheets.add(name);sheets[name]=s;s.showGridLines=false;
  s.getRange('A2').values=[[title]];
  s.getRange('A2').format.font={name:'Arial',size:14,bold:true,color:'#24354B'};
  s.getRangeByIndexes(5,0,1,heads.length).values=[heads];
  s.getRangeByIndexes(6,0,rows.length,heads.length).values=rows.map(r=>r.map(x=>x==null?'n.a.':x));
  const used=s.getRangeByIndexes(5,0,rows.length+1,heads.length);
  used.format.font={name:'Arial',size:10,color:'#202733'};
  used.format.verticalAlignment='center';used.format.rowHeight=25;
  const hdr=s.getRangeByIndexes(5,0,1,heads.length);hdr.format={fill:'#334C67',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:40,horizontalAlignment:'center',verticalAlignment:'center'};
  widths.forEach((w,i)=>s.getRangeByIndexes(0,i,rows.length+6,1).format.columnWidth=w);
  const table=s.tables.add(`A6:${col(heads.length)}${rows.length+6}`,true,'AtlasTable'+Object.keys(sheets).length);
  table.showFilterButton=true;
  s.freezePanes.freezeRows(6);s.freezePanes.freezeColumns(1);
  return s;
}
function col(n){let a='';while(n){n--;a=String.fromCharCode(65+n%26)+a;n=Math.floor(n/26);}return a;}
const reader=wb.worksheets.add('加载时长与峰位');sheets['加载时长与峰位']=reader;reader.showGridLines=false;reader.tabColor='#334C67';
reader.getRange('A2').values=[['TPMS历史模拟：加载时长与峰位']];reader.getRange('A2').format.font={name:'Arial',size:14,bold:true,color:'#24354B'};
reader.getRange('A4:B4').values=[['旧固定分母 (N)',2.2589142322540283]];
reader.getRange('B4').setNumberFormat('0.000000');
const runFields=['id','case','solver','T_ms','t_mm','N','element','quadrature_points','occupancy','interface_width_mm','kernel','a_start','a_end','recorded_max_at','recorded_max_N','interior_peak_observed','end_loading_force_N','hold_mean_N','status','role','original_failed_checks','body_minutes','note','source'];
const rheads=['曲线ID','构型','求解方式','T (ms)','厚度 (mm)','N','单元','积分点数','占据','界面宽 (mm)','材料核','记录起点 (%)','记录末点 (%)','加载最大力位置 (%)','加载最大力 (N)','内部峰有后续降载','加载末点力 (N)','保载均力 (N)','覆盖/停止状态','记录用途','原失败检查','主体性能计时 (min)','说明','原数据来源'];
const rs=runs.map(r=>runFields.map(k=>k==='interior_peak_observed'?(r[k]?'已观察':'未观察'):k.startsWith('a_')||k==='recorded_max_at'?100*r[k]:k==='source'?'/home/xuehu/projects/tpms_jax/'+r[k]:r[k]));
const catalog=tab('曲线条件','全部有效曲线和短段的原条件',rheads,rs,[17,17,20,12,12,8,18,12,12,13,15,15,15,20,18,18,18,18,24,20,60,22,70,100]);
catalog.getRange(`D7:E${runs.length+6}`).setNumberFormat('0.000');catalog.getRange(`J7:O${runs.length+6}`).setNumberFormat('0.0000');catalog.getRange(`Q7:R${runs.length+6}`).setNumberFormat('0.0000');catalog.getRange(`V7:V${runs.length+6}`).setNumberFormat('0.00');
const rowid=new Map(runs.map((r,i)=>[r.id,i+7]));
const shell=runs.filter(r=>r.role==='壳速率').sort((a,b)=>a.case.localeCompare(b.case)||b.T_ms-a.T_ms);
reader.getRange('A6:H6').values=[['曲线ID','构型','T (ms)','最大力位置 (%)','加载最大力 (N)','对40ms峰位差 (百分点)','对40ms峰力差 (%)','原筛查范围']];
reader.getRange('A6:H6').format={fill:'#334C67',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:40,horizontalAlignment:'center'};
for(let i=0;i<shell.length;i++){
  const r=shell[i],row=i+7,cr=rowid.get(r.id),base=rowid.get(shell.find(s=>s.case===r.case&&s.T_ms===40).id);
  reader.getRange(`A${row}:H${row}`).values=[[r.id,r.case,r.T_ms,null,null,null,null,r.original_failed_checks?'原能量/惯性等有失败，详见曲线条件':'原检查无失败']];
  reader.getRange(`D${row}:G${row}`).formulas=[[`='曲线条件'!N${cr}`,`='曲线条件'!O${cr}`,`=D${row}-'曲线条件'!N${base}`,`=E${row}/'曲线条件'!O${base}-1`]];
}
reader.getRange('A7:H13').format.font={name:'Arial',size:10,color:'#202733'};reader.getRange('A7:H13').format.rowHeight=27;
reader.getRange('C7:F13').setNumberFormat('0.0000');reader.getRange('G7:G13').setNumberFormat('0.00%');
[18,17,12,20,20,24,22,48].forEach((w,i)=>reader.getRangeByIndexes(0,i,36,1).format.columnWidth=w);
const notes=[
 ['数据范围','只汇总已有diverse_04/28记录；41份去重曲线/短段、22个静力点。没有新增仿真。'],
 ['T的含义','Explicit中T是20%幅值smoothstep的加载时长，单位ms。Standard步时间是伪时间。'],
 ['最大力与峰','加载最大力按已保存加载记录取最大；末端仍增载时，不能声称已捕获峰。'],
 ['压缩与反力','宏观压缩量为轴向缩短/10mm；反力以压缩为正，保留真实负向振荡。'],
 ['RMS误差','共同窗口均匀2001点的反力差均方根，单位N；再除参考完整加载最大力。'],
 ['两种分母','保留原固定2.258914N分母；另列各参考完整加载最大力。不能混用百分比。'],
 ['功差','功=10mm×∫反力d压缩量；原节点及边界插值积分。与参考在同覆盖窗口比较。'],
 ['定点力差','(本次力/参考力−1)；参考力很小时可很大，不能替代曲线RMS。'],
 ['保载均力','各实际保载时窗的时间积分均值。不同T的保载时间也不同，不直接归因网格。'],
 ['范围缺口','未覆盖记n.a.；不把拒绝态、预算前缀、局部重放拼成完整路径。'],
 ['静力K','K=反力/轴向位移，单位N/mm；初始压缩0.01%对应0.001mm，无惯性。'],
 ['表格用途','统一回顾指标，不覆盖原验收判据。约10%是工作目标，原失败检查仍保留。'],
 ['重新计算','复杂积分/RMS在只读Python中计算；表内比值公式随输入值更新，不调用求解器。'],
 ['共同物理量','L10mm、E10MPa、ν0.3、实体密度10⁻⁹ tonne/mm³；背景η10⁻⁴，XYZ横向宏观应变0。'],
 ['历史例外','旧平端诊断、厚度扰动、填料嵌入、准备探针与局部重放已分别注明，不能混作生产结果。'],
];
reader.getRange('A17:B31').values=notes;reader.getRange('A17:B31').format.font={name:'Arial',size:10,color:'#374151'};reader.getRange('A17:B31').format.rowHeight=26;
reader.getRange('A17:A31').format.font={name:'Arial',size:10,bold:true,color:'#24354B'};

const eh=['曲线ID','构型','参考ID','窗口','起点 (%)','末点 (%)','同T','同厚度','本T (ms)','参考T (ms)','RMS力差 (N)','参考完整加载最大力 (N)','RMS/参考最大力 (%)','RMS/旧分母 (%)','本功 (N mm)','参考功 (N mm)','相对功差 (%)','窗口末力 (N)','参考末力 (N)','末点相对力差 (%)','范围说明'];
const er=errors.map(e=>[e.id,e.case,e.reference_id,e.window,100*e.a_start,100*e.a_end,e.same_T?'是':'否',e.same_thickness?'是':'否',e.T_ms,e.reference_T_ms,e.RMS_N,e.reference_full_loading_max_N,null,null,e.work_N_mm,e.reference_work_N_mm,null,e.end_force_N,e.reference_end_force_N,null,e.scope]);
const es=tab('窗口误差','完整与部分路径的共同窗口误差',eh,er,[17,17,17,16,13,13,10,10,12,12,18,22,23,21,19,19,20,19,19,23,70]);
for(let i=0;i<er.length;i++){let r=i+7;es.getRange(`M${r}:N${r}`).formulas=[[`=K${r}/L${r}`,`=K${r}/'加载时长与峰位'!$B$4`]];es.getRange(`Q${r}`).formulas=[[`=O${r}/P${r}-1`]];es.getRange(`T${r}`).formulas=[[`=R${r}/S${r}-1`]];}
es.getRange(`E7:F${er.length+6}`).setNumberFormat('0.0000');es.getRange(`I7:L${er.length+6}`).setNumberFormat('0.000000');es.getRange(`M7:N${er.length+6}`).setNumberFormat('0.00%');es.getRange(`O7:P${er.length+6}`).setNumberFormat('0.000000');es.getRange(`Q7:Q${er.length+6}`).setNumberFormat('0.00%');es.getRange(`R7:S${er.length+6}`).setNumberFormat('0.000000');es.getRange(`T7:T${er.length+6}`).setNumberFormat('0.00%');
const ph=['曲线ID','构型','参考ID','压缩量 (%)','本力 (N)','参考力 (N)','差值 (N)','定点相对力差 (%)','差值/参考加载最大力 (%)','同T','数据类型'];
const pr=points.map(p=>[p.id,p.case,p.reference_id,100*p.compression,p.force_N,p.reference_force_N,null,null,p.difference_over_reference_max,p.same_T?'是':'否',p.point_kind]);
const ps=tab('定点误差','固定压缩量和已有局部记录的反力差',ph,pr,[17,17,17,18,18,18,18,24,28,10,30]);
for(let i=0;i<pr.length;i++){let r=i+7;ps.getRange(`G${r}:H${r}`).formulas=[[`=E${r}-F${r}`,`=E${r}/F${r}-1`]];}
ps.getRange(`D7:G${pr.length+6}`).setNumberFormat('0.000000');ps.getRange(`H7:I${pr.length+6}`).setNumberFormat('0.00%');
const peh=['曲线ID','构型','参考ID','同T','同厚度','两者均观察内部峰','峰位差 (百分点)','峰力相对差 (%)','本保载均力 (N)','参考保载均力 (N)','保载均力相对差 (%)','保载比较范围','本覆盖状态'];
const per=paths.map(p=>[p.id,p.case,p.reference_id,p.same_T?'是':'否',p.same_thickness?'是':'否',p.peak_observed?'是':'否',p.peak_position_difference_pp,p.peak_force_relative,p.hold_mean_N,p.reference_hold_mean_N,null,p.hold_scope,p.coverage]);
const pes=tab('峰与保载差','观察峰和已完成保载的对照',peh,per,[17,17,17,10,10,22,21,24,24,24,30,75,35]);
for(let i=0;i<per.length;i++){let r=i+7;if(paths[i].hold_mean_relative!=null)pes.getRange(`K${r}`).formulas=[[`=I${r}/J${r}-1`]];else pes.getRange(`K${r}`).values=[['n.a.']];}
pes.getRange(`G7:G${per.length+6}`).setNumberFormat('0.0000');pes.getRange(`H7:H${per.length+6}`).setNumberFormat('0.00%');pes.getRange(`I7:J${per.length+6}`).setNumberFormat('0.000000');pes.getRange(`K7:K${per.length+6}`).setNumberFormat('0.00%');
const sh=['点ID','构型','条件','N','单元','积分规则','厚度 (mm)','界面宽 (mm)','占据','压缩量 (%)','本力 (N)','K/割线K (N/mm)','匹配壳K (N/mm)','匹配壳力 (N)','相对壳 (%)','边界','状态','范围','本来源','参考来源'];
const sr=statics.map(p=>[p.id,p.case,p.label,p.N,p.element,p.quadrature,p.t_mm,p.width_mm,p.occupancy,100*p.compression,p.force_N,p.K_N_per_mm,p.shell_K_N_per_mm,p.shell_force_N,null,p.PBC,p.status,p.scope,p.source,p.reference_source]);
const ss=tab('初始与静力点','初始K、早期有限应变平衡和厚度扰动',sh,sr,[25,17,35,8,13,24,15,18,12,18,20,25,25,22,20,22,18,65,90,90]);
for(let i=0;i<sr.length;i++){let r=i+7,p=statics[i];if(p.shell_K_N_per_mm!=null)ss.getRange(`O${r}`).formulas=[[`=L${r}/M${r}-1`]];else if(p.shell_force_N!=null)ss.getRange(`O${r}`).formulas=[[`=K${r}/N${r}-1`]];else ss.getRange(`O${r}`).values=[['n.a.']];}
ss.getRange(`G7:H${sr.length+6}`).setNumberFormat('0.000');ss.getRange(`J7:N${sr.length+6}`).setNumberFormat('0.000000');ss.getRange(`O7:O${sr.length+6}`).setNumberFormat('0.00%');
tab('记录范围','重复副本、无有效路径和原失败记录', ['来源','原ID','保留原因/不能画为独立路径的原因'], coverage.map(c=>[c.source,c.id,c.reason]),[95,24,120]);

wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'加载时长与峰位!A6:G13',include:'values,formulas',tableMaxRows:8,tableMaxCols:7,maxChars:3500})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},maxChars:2000})).ndjson);
for(const [name,range] of [['加载时长与峰位','A1:H14'],['加载时长与峰位','A16:H31'],['曲线条件','A5:M12'],['窗口误差','A5:N12'],['定点误差','A5:K12'],['峰与保载差','A5:K12'],['初始与静力点','A5:O12'],['记录范围','A5:C12']]){
  const blob=await wb.render({sheetName:name,range,scale:1.4,format:'png'});
  await fs.writeFile(path.join(import.meta.dirname,'preview_'+name+'_'+range.replace(':','-')+'.png'),new Uint8Array(await blob.arrayBuffer()));
}
const xlsx=await SpreadsheetFile.exportXlsx(wb);await xlsx.save(path.join(out,'TPMS_历史模拟条件与误差.xlsx'));
console.log('XLSX exported');
} catch(e) { console.error(e.message); process.exitCode=1; }
