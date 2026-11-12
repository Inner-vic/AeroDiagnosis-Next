"""Produce a revised thesis chapter without modifying the archived original."""
from pathlib import Path
import hashlib,json,re,shutil
H=Path(__file__).resolve().parent
OLD=Path('D:/codex/worktrees/knowledge-upgrade-integration/AeroDiagnosis/docs/upgrade/chapter5-draft')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
text=(OLD/'CHAPTER-5-DRAFT.md').read_text(encoding='utf8')
text=text.replace('## 5.4 门控混合检索与证据约束回答','## 5.4 基于门控与路径校验的混合检索及证据约束回答')
anchor='后续将通过检索组合对比、分题型评价和消融实验分析所提方法的适用条件，并结合实际资料问答案例验证系统功能。'
text=text.replace(anchor,anchor+'本章围绕以下问题展开：混合文本检索能否形成可靠的知识查询基础；图谱能否补全跨片段关联所需的证据；参与门控能否减少对原有正确证据的干扰。研究重点为来源关联的路径证据组织及其受约束使用，不预设检索通道数量与效果之间存在单调关系。')
text=text.replace('### 5.5.3 单路、双路与三路检索对比','### 5.5.3 文本检索基础与多通道比较')
text=text.replace('### 5.5.4 分题型与消融分析','### 5.5.4 图谱证据补全、干扰及消融分析')
text=text.replace('设一条来源关系记录为','图谱用于组织资料中明确记载的关系，不能把抽取得到的每条边自动视为真实因果规律。设一条来源关系记录为')
metrics=r'''
为直接衡量图谱的补证收益及其干扰，以相同问题、来源范围和证据预算下的双文本检索为配对基线。令 \(b_q\) 与 \(h_q\) 分别表示基线和待比较方法在前\(K\)条证据中是否获得至少一个完整的必要证据集合。定义证据补全率与正确证据受损率为

\[
\mathrm{ECR}@K=\frac{\sum_q(1-b_q)h_q}{\sum_q(1-b_q)},\qquad
\mathrm{EDR}@K=\frac{\sum_q b_q(1-h_q)}{\sum_q b_q}. \tag{5-19}
\]

ECR的分母为基线证据不完整的题数，EDR的分母为基线证据完整的题数。分母为0时记为不适用，并报告原始计数，不将其写成0%或100%。本章所称“正确证据受损”特指完整证据由完整变为不完整，不能代替答案正确性或排序质量评价；完整性未变而排序下降的情况仍由nDCG等指标反映。补全率也不自动等于图谱的纯贡献，需要结合图谱触发、图谱新增片段以及相同候选预算的文本对照判断。对存在多个等价答案依据的问题，任一预先标注的充分证据集合完整命中即记为完整。

除两个条件比例外，同时报告改善题数\(N_{01}\)、退化题数\(N_{10}\)、持平题数及净改善\(N_{01}-N_{10}\)。指标分别在受控资料和真实来源资料上统计，不混合汇总。题目与证据标注、关系路径和方法参数均应在新测试执行前固定，不能依据检索结果补改参考证据以形成预期排序。
'''
needle='### 5.5.3 文本检索基础与多通道比较'
text=text.replace(needle,metrics+'\n'+needle)
audit=json.loads((H/'paired-evidence-audit.json').read_text(encoding='utf8'))
names={'BDG_scope':'普通三路（独立调优）','BDG_scope_matched':'普通三路（匹配权重）','BDG_gated_scope':'门控三路','BDG_atomic_scope':'完整路径插入'}
table='\n表5-11 历史受控集的配对证据补全与受损统计（K=5）\n\n| 方法 | 补全题数/基线不完整题数 | 补全率 | 受损题数/基线完整题数 | 受损率 | 净改善题数 |\n|---|---:|---:|---:|---:|---:|\n'
for method,name in names.items():
 a=audit['comparisons'][f'BD_scope -> {method} @ 5']['ALL']
 table+=f"| {name} | {a['improved']}/{a['baseline_incomplete']} | {a['completion_rate']:.2%} | {a['damaged']}/{a['baseline_complete']} | {a['damage_rate']:.2%} | {a['net_gain']} |\n"
table+='\n上述结果重新说明了补全与干扰之间的区别。匹配权重普通三路补齐34题，同时损伤36道原本完整的题，净变化为−2题；门控方案补齐34题且未损伤原本完整的182题。两个方案均具有100%的条件补全率，但总体效果不同。该复算使用已见历史结果，不能作为新增独立验证，也不能据零退化推断部署后没有风险。完整路径插入具有相同的完整@5，但其排序指标仍需结合表5-7分析。\n'
text=text.replace('### 5.5.5 端到端回答实验',table+'\n### 5.5.5 端到端回答实验')
# Eliminate an ambiguous duplicate label in the existing ablation table.
text=text.replace('| BM25＋E5 | 150/216 | 156/216 | 170/216 |','| BM25＋E5（全库） | 150/216 | 156/216 | 170/216 |')
text=text.replace('| BM25＋E5 | 164/216 | 182/216 | 182/216 |','| BM25＋E5（来源约束） | 164/216 | 182/216 | 182/216 |')
# Keep headings and table numbers in reading order.
text=text.replace('表5-11','表5-8').replace('表5-8所示','表5-9所示') if False else text
for n in [10,9,8]:text=text.replace(f'表5-{n}',f'表5-{n+1}')
text=text.replace('表5-11 历史受控集','表5-8 历史受控集')
# Resolve original images relative to this new file without touching source assets.
text=re.sub(r'!\[([^]]*)\]\(([^)]+)\)',lambda m:f'![{m[1]}]({(OLD/m[2]).resolve().as_posix()})',text)
# References are the previously verified list, not newly invented citations.
ref=Path('docs/upgrade/chapter5-word/CHAPTER-5-WORD-SOURCE.md').read_text(encoding='utf8')
match=re.search(r'^##? 参考文献.*',ref,re.M|re.S)
if match:text+='\n\n'+match[0]
(H/'CHAPTER-5-REVISED.md').write_text(text,encoding='utf8')
(H/'revision-audit.json').write_text(json.dumps({'date':'2026-11-12','original':str(OLD/'CHAPTER-5-DRAFT.md'),'original_sha256':sha(OLD/'CHAPTER-5-DRAFT.md'),'revised_sha256':sha(H/'CHAPTER-5-REVISED.md'),'paid_calls':0,'original_preserved':True,'new_independent_results_inserted':False},ensure_ascii=False,indent=2),encoding='utf8')
print('Revised chapter written; original preserved')
