"""Descriptive manuscript draft figure; no journal-specific compliance claim."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

root=Path('D:/codex/worktrees/knowledge-upgrade-integration/AeroDiagnosis')
source=root/'.runtime/controlled-scenario-v1/test-summary.json'
out=root/'docs/upgrade/execution/figures/controlled-scenario-v1'
out.mkdir(parents=True,exist_ok=True)
s=json.loads(source.read_text(encoding='utf8'))
methods=list(s)
labels=['BM25','E5','KG','BM25+E5','BM25+KG','E5+KG','BM25+E5+KG（路径）','BM25+E5+KG（单边）']
kinds=['exact','paraphrase','conditions','chain2','chain3','boundary']
kind_labels=['精确匹配','语义改述','条件组合','两边链','三边链','逆推边界']
fontfile=Path('C:/Windows/Fonts/msyh.ttc')
font=FontProperties(fname=str(fontfile))
family=font.get_name()
values=np.array([[s[m][k]['at']['5']['complete'] for k in kinds] for m in methods])
overall=np.array([s[m]['ALL']['at']['5']['complete'] for m in methods])
data=dict(methods=methods,kinds=kinds,overall_complete5=overall.tolist(),stratified_complete5=values.tolist(),per_type_n=24,total_questions=144,template_clusters=8,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
(out/'figure-data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
with plt.rc_context({'font.family':family,'font.size':10,'axes.unicode_minus':False,'svg.fonttype':'none','savefig.facecolor':'white'}):
    fig,(ax,heat)=plt.subplots(1,2,figsize=(14,6.6),layout='constrained',gridspec_kw={'width_ratios':[1.15,1.4]})
    y=np.arange(len(methods))
    ax.barh(y,overall*100,color='#366A91',height=0.6,edgecolor='#18374E')
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,100)
    ax.set_xlabel('Top5必要证据完整率（%）');ax.set_title('A  全部留出测试题（n=144）',loc='left')
    ax.grid(axis='x',alpha=0.2);ax.set_axisbelow(True)
    for i,m in enumerate(methods):ax.text(overall[i]*100+1,i,f'{round(overall[i]*144)}/144',va='center',fontsize=9)
    image=heat.imshow(values*100,vmin=0,vmax=100,cmap='cividis',aspect='auto',interpolation='nearest')
    heat.set_xticks(np.arange(6),kind_labels,rotation=25,ha='right')
    heat.set_yticks(y,labels);heat.tick_params(axis='y',labelsize=8)
    heat.set_title('B  分型完整率（每类n=24，%）',loc='left')
    contrasts=[]
    for i in range(len(methods)):
        for j in range(6):
            rgb=image.cmap(image.norm(values[i,j]*100))[:3]
            linear=[c/12.92 if c<=0.04045 else ((c+0.055)/1.055)**2.4 for c in rgb]
            luminance=sum(c*w for c,w in zip(linear,[.2126,.7152,.0722],strict=True))
            black=(luminance+.05)/.05;white=1.05/(luminance+.05)
            color='black' if black>=white else 'white';contrasts.append(max(black,white))
            heat.text(j,i,f'{values[i,j]*100:.0f}',ha='center',va='center',color=color,fontsize=10)
    fig.colorbar(image,ax=heat,label='完整率（%）',fraction=.04,pad=.025)
    fig.suptitle('自建航空发动机场景评测：受控规则世界，8个留出机理模板',fontsize=14)
    fig.supxlabel('图谱为已知因果规则子集；不覆盖参数、条件与边界。两链题共享前缀，不能将144题视作144个独立实测案例。\n柱形与色阶从0起；图为固定测试集描述统计，不表示现实维修总体精度。',fontsize=9)
    fig.savefig(out/'results.png',dpi=300)
    fig.savefig(out/'results.svg')
    plt.close(fig)
manifest=dict(source=str(source),source_sha256=data['source_sha256'],matplotlib=matplotlib.__version__,width_inches=14,height_inches=6.6,dpi=300,font=family,font_file=str(fontfile),normalization='complete evidence@5 proportion *100; no exclusions or smoothing',uncertainty='descriptive fixed test counts only; paired template-bootstrap interval separately in report',template_clusters=8,min_cell_text_contrast=min(contrasts),journal_requirements='Not specified; provisional general manuscript figure, verify publisher requirements before submission',alt_text_zh='八方法总体Top5完整率从47.22%到68.06%。三路68.06%、双文本66.67%。条件组合各方法均零；三边链图谱单路100%、三路8.33%。全部方法和零值均展示。',files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()})
(out/'figure-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(output=str(out),minimum_cell_text_contrast=min(contrasts)),ensure_ascii=False))
