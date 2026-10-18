from pathlib import Path
import json,re,hashlib,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
R=Path('D:/codex/worktrees/knowledge-upgrade-integration/AeroDiagnosis')
C=R/'docs/upgrade/chapter5-draft'
F=C/'figures'
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'svg.fonttype':'none','font.family':font.get_name(),'axes.unicode_minus':False})
fig,axs=plt.subplots(1,2,figsize=(10,5),sharey=True)
data=[('实际FAA资料：12道新问题',[11,11],['91.67%\n11/12','91.67%\n11/12']),('受控三边链：12道已见模板问题',[1,12],['8.33%\n1/12','100%\n12/12'])]
for ax,(title,counts,labels) in zip(axs,data):
 bars=ax.bar(['BM25＋向量','门控三路'],[x/12*100 for x in counts],color=['#557a9a','#397c69'],width=.55)
 ax.set_title(title,fontsize=11,pad=16);ax.set_ylim(0,115);ax.set_yticks([0,20,40,60,80,100]);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 for b,label in zip(bars,labels):ax.text(b.get_x()+b.get_width()/2,b.get_height()+2,label,ha='center',va='bottom',fontsize=10)
 ax.spines[['top','right']].set_visible(False)
axs[0].set_ylabel('任务判定正确率（%）')
fig.text(.5,.045,'两组分别统计；实际组含4道不可回答题。来源及回答经独立AI非专家复核。',ha='center',fontsize=9)
fig.text(.5,.01,'受控组为程序化规则场景，结果不代表真实发动机诊断准确率；本图仅作描述性比较。',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.10,1,1])
for ext in ['png','svg']:fig.savefig(F/f'figure-5-6-answer-results.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig)
shutil.copyfile(__file__,F/'draw_chapter5_answer_results.py')
(F/'figure-5-6-data.json').write_text(json.dumps({'groups':[{'id':'real_faa','n_per_method':12,'correct':[11,11],'answerable':8,'unanswerable':4},{'id':'controlled_seen_chain3','n_per_method':12,'correct':[1,12],'answerable':12,'unanswerable':0}],'methods':['BD','GATED'],'review':'blinded independent AI nonexpert','uncertainty':'descriptive only; no pooled accuracy or independence assumption','matplotlib':matplotlib.__version__},ensure_ascii=False,indent=2),encoding='utf-8')
for name in ['CHAPTER-5-SECTIONS-5.5-5.6.md','CHAPTER-5-DRAFT.md']:
 p=C/name
 b=p.read_text(encoding='utf-8')
 # Repair the LaTeX commands damaged by interpolation in the initial results writer.
 start=b.index('### 5.5.2 ');end=b.index('### 5.5.3 ',start)
 section=b[start:end]
 section=section.replace('\t'+'ag{',r'\tag{')
 for a,z in [('mathrm',r'\mathrm'),('sum_',r'\sum_'),('mathbf',r'\mathbf'),('subseteq',r'\subseteq'),('cap ',r'\cap '),('qquad',r'\qquad'),('^ast',r'^\ast')]:
  section=section.replace(a,z)
 section=re.sub(r'(?m)^\[$',lambda _:r'\[',section)
 section=re.sub(r'(?m)^\]$',lambda _:r'\]',section)
 for a,z in [(' (S_q)',r' \(S_q\)'),(' (K)',r' \(K\)'),(' (R_q^K)',r' \(R_q^K\)'),(' (r_q^\\ast)',r' \(r_q^\ast\)'),(' (S_q\\subseteq R_q^{r_q^\\ast})',r' \(S_q\subseteq R_q^{r_q^\ast}\)'),(' (J_q)',r' \(J_q\)')]:section=section.replace(a,z)
 b=b[:start]+section+b[end:];p.write_text(b,encoding='utf-8')
body=(C/'CHAPTER-5-DRAFT.md').read_text(encoding='utf-8')
refs=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',body)
missing=[p for p in refs if not (C/p).exists()]
assert not missing,missing
tags=[int(x) for x in re.findall(r'\\tag\{5-(\d+)\}',body)]
assert tags==list(range(1,19)),tags
assert all(f'## 5.{i} ' in body for i in range(1,7))
notes=C/'METHODS-WRITING-NOTES.md'
text=notes.read_text(encoding='utf-8').replace('5.5与5.6尚未在本稿展开。','此为5.2—5.4阶段记录；后续已完成5.5、5.6及实际配对问答实验，见完整稿和EXPERIMENT-WRITING-NOTES.md。')
notes.write_text(text,encoding='utf-8')
contract=R/'docs/upgrade/thesis-chapter-contract.json'
j=json.loads(contract.read_text(encoding='utf-8-sig'));j['contract_version']='3.3';j['budget_breakdown_cny']={'settled_upper_estimate':40.940237,'unknown_charge_reservations':.92103,'remaining_unallocated':.138733}
j['current_turn']='Completed 48 paired answers on 24 questions, source-reviewed and method-blinded AI nonexpert scored; FAA new questions on seen sources BD=GATED11/12; controlled seen chain3 BD1/12 GATED12/12. No pooling or independent holdout claim. Added5.5-5.6, full chapter18 equations10 tables7 figures.72 new provider requests cost upper0.383919CNY, cumulative41.861267/42; no new pending calls.'
j['chapter5_answer_experiment']={'path':'docs/upgrade/chapter5-draft/experiments/answer-eval-v1','report':'docs/upgrade/execution/CHAPTER5-ANSWER-EVALUATION-2026-10-17.md','expert_reviewed':False,'new_source_holdout':False,'controlled_templates_seen':True}
contract.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
plan=R/'docs/upgrade/THESIS-CHAPTER-DETAILED-PLAN.md'
with plan.open('a',encoding='utf-8') as f:f.write('\n\n## 40 第五章端到端实验与结果正文（2026年10月17日）\n\n实际执行24题48份配对回答，冻结后未改题、改标签或删失败，来源和回答由独立AI非专家复核，评分隐藏方法名称。FAA已有来源的新12题两方法均11/12，4/4证据不足题安全拒答；门控均回退文本，无额外图谱收益。已见受控三边链12题双文本1/12、门控三路12/12，仅说明规则场景完整路径作用。两组分开统计，保留共同误拒答案例；134条最终引用身份/哈希/偏移检查通过，不将保留陈述支持率解释为原始生成零幻觉。完成5.5、5.6及新结果图，完整稿含18公式、10表、7图。72次新增请求保守0.383919元，累计41.861267/42元，无本轮待决请求。详见[实验报告](execution/CHAPTER5-ANSWER-EVALUATION-2026-10-17.md)。\n')
report='''# 第五章新增端到端实验报告

日期：2026年10月17日。已执行，不是实验计划。

|组别|问题性质|BM25＋向量|门控三路|配对变化|
|---|---|---:|---:|---|
|FAA资料12题|已有公开来源的新问题，8题有答案、4题不可回答|11/12（91.67%）|11/12（91.67%）|0改善、0退化、12持平|
|受控三边链12题|程序化规则场景、已见模板，全部有答案|1/12（8.33%）|12/12（100%）|11改善、0退化、1持平|

实际组门控0/12开启，因此未观察到额外图谱收益；两方法有答案题完整回答7/8，安全拒答4/4。共同失败cq-02为资料范围解释过于保守的拒答，按冻结标签保留失败，需后续专业审核。受控组12/12开启门控并保留完整路径，说明该场景必要证据补充与回答完整性有关，不能推广为真实故障诊断准确率。

问题、参考证据和代码在生成前冻结；两种方法交替顺序执行，相同模型、提示流程、输出上限和证据预算。48份输出无删除和重跑。独立复核隐藏方法身份，审核者为AI非专家。两组来源分布与问题性质不同，禁止合并正确率。实际资料不是新来源独立测试，受控模板已见，结果作描述性能力验证。

134条最终引用通过来源身份、文本哈希和偏移核验；125条最终保留陈述经AI复核具有支持。这不是对过滤前模型幻觉率的测量。缓存命中会影响时间和分方法费用，不能据此声称成本优势。

96次逻辑模型调用中72次新增请求、24次精确输入缓存命中。新增费用保守上界0.383919元，累计41.861267/42元；含历史待决预留0.921030元，无本轮新增待决调用。不再追加本轮付费实验。

[完整第五章](../chapter5-draft/CHAPTER-5-DRAFT.md)包含5.5的指标公式、完整单路/双路/三路与消融表、分题型结果、时延和失败分析，以及5.6小结。原有19配置4104条检索排名为此前冻结实验，本轮未冒充新独立复测。

[逐题预测、匿名评分和冻结记录](../chapter5-draft/experiments/answer-eval-v1/summary.json)；同目录保留完整预测、来源、审核、审计和执行脚本。图5-6绘图数据与脚本位于正文figures目录。正式文献核验与学校Word排版仍待完成。
'''
(R/'docs/upgrade/execution/CHAPTER5-ANSWER-EVALUATION-2026-10-17.md').write_text(report,encoding='utf-8')
files=[p for p in C.rglob('*') if p.is_file() and p.name!='final-experiment-writing-audit.json']
audit={'date':'2026-10-17','sections':[f'5.{i}' for i in range(1,7)],'equation_tags':tags,'image_links':refs,'all_image_links_resolve':True,'files_sha256':{str(p.relative_to(C)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'scope':'24 paired questions,48 answers; separate real seen-source and controlled seen-template groups; no new paid calls during report generation'}
(C/'final-experiment-writing-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'equations':len(tags),'images':len(refs),'hashed_files':len(files),'chapter_characters':len(body)}))
