from pathlib import Path
import json,hashlib,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
O=Path('D:/codex/worktrees/knowledge-upgrade-integration/AeroDiagnosis/docs/upgrade/execution/gated-triple-paper-2026-10-16')
s=json.loads((O/'all-method-summary.json').read_text(encoding='utf-8'))
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'svg.fonttype':'none','font.size':10})
methods=['B_scope','D_scope','G_scope','BD_scope','BG_scope','DG_scope','BDG_scope_matched','BDG_gated_scope']
labels=['BM25*','中文E5*','图谱*','BM25＋E5','BM25＋图谱*','E5＋图谱*','普通三路','门控三路']
values=[s[m]['ALL']['at']['5']['complete']*100 for m in methods]
fig,ax=plt.subplots(figsize=(9,5.3),layout='constrained')
ax.barh(np.arange(8),values,color=['#555555']*7+['#006B53'],height=.64)
ax.set_yticks(np.arange(8),labels);ax.invert_yaxis();ax.set_xlim(0,100)
ax.set_xlabel('完整必要证据@5（%）；216题，12个模板')
ax.set_title('统一来源约束下的检索组合比较')
for i,v in enumerate(values):
 ax.text(v-1.2,i,f'{v:.2f}% ({round(v*216/100)}/216)',ha='right',va='center',color='white',fontweight='bold')
ax.spines[['top','right']].set_visible(False)
fig.savefig(O/'overall-scoped.png',dpi=220);fig.savefig(O/'overall-scoped.svg');plt.close(fig)
kinds=['exact','paraphrase','conditions','chain2','chain3','boundary']
names=['参数事实','同义改写','条件组合','两边因果链','三边因果链','不确定性边界']
data=np.array([[s[m][k]['at']['5']['complete']*100 for m in methods] for k in kinds])
fig,ax=plt.subplots(figsize=(11.3,5.1),layout='constrained')
im=ax.imshow(data,vmin=0,vmax=100,cmap='Blues',aspect='auto')
ax.set_xticks(range(8),labels,rotation=22,ha='right');ax.set_yticks(range(6),names)
ax.set_title('分题型完整必要证据@5（每类36题）')
for i in range(6):
 for j in range(8):ax.text(j,i,f'{round(data[i,j]*36/100)}/36',ha='center',va='center',color='white' if data[i,j]>=65 else 'black')
fig.colorbar(im,ax=ax,label='完整证据率（%）',shrink=.8)
fig.savefig(O/'type-scoped.png',dpi=220);fig.savefig(O/'type-scoped.svg');plt.close(fig)
shutil.copyfile(__file__,O/'plot_gated_paper.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(O/'figure-manifest.json').write_text(json.dumps({'matplotlib':matplotlib.__version__,'input_sha256':sha(O/'all-method-summary.json'),'methods':methods,'zero_baseline':True,'heatmap_scale':[0,100],'uncertainty':'descriptive empirical proportions; no independent-binomial intervals','publisher':'unspecified thesis draft','figures':{p.name:sha(p) for p in O.glob('*.png')}},ensure_ascii=False,indent=2),encoding='utf-8')
print(O)
