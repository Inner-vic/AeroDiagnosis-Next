from pathlib import Path
import json,hashlib,shutil
from html import escape
from PIL import Image,ImageDraw,ImageFont
R=Path('D:/codex/worktrees/knowledge-upgrade-integration/AeroDiagnosis')
O=R/'docs/upgrade/chapter5-draft/figures';O.mkdir(parents=True,exist_ok=True)
FONT='C:/Windows/Fonts/msyh.ttc'
class Diagram:
 def __init__(self,w,h,title):
  self.w,self.h=w,h;self.im=Image.new('RGB',(w,h),'white');self.d=ImageDraw.Draw(self.im)
  self.svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="white"/><defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#34434c"/></marker></defs>']
  self.text(w/2,30,title,26)
 def text(self,x,y,text,size=23,fill='#17232b'):
  font=ImageFont.truetype(FONT,size)
  for i,line in enumerate(text.split('\n')):
   yy=y+(i-(len(text.split('\n'))-1)/2)*(size+10)
   self.d.text((x,yy),line,font=font,fill=fill,anchor='mm')
   self.svg.append(f'<text x="{x}" y="{yy}" text-anchor="middle" dominant-baseline="central" font-family="Microsoft YaHei, sans-serif" font-size="{size}" fill="{fill}">{escape(line)}</text>')
 def box(self,x,y,w,h,text,fill='white',size=23):
  self.d.rounded_rectangle((x,y,x+w,y+h),radius=6,fill=fill,outline='#34434c',width=2)
  self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="#34434c" stroke-width="2"/>')
  self.text(x+w/2,y+h/2,text,size)
 def edge(self,points,label='',at=None,dash=False):
  self.d.line(points,fill='#34434c',width=2)
  import math
  x,y=points[-1];a,b=points[-2];angle=math.atan2(y-b,x-a)
  triangle=[(x,y),(x-11*math.cos(angle-.45),y-11*math.sin(angle-.45)),(x-11*math.cos(angle+.45),y-11*math.sin(angle+.45))]
  self.d.polygon(triangle,fill='#34434c')
  # SVG preserves the semantic dashed connector used for task-level coupling.
  if dash:
   for a,b in zip(points,points[1:]):
    self.d.line([a,b],fill='white',width=3)
    n=max(1,int(math.dist(a,b)/8))
    for i in range(0,n,2):self.d.line([(a[0]+(b[0]-a[0])*i/n,a[1]+(b[1]-a[1])*i/n),(a[0]+(b[0]-a[0])*min(i+1,n)/n,a[1]+(b[1]-a[1])*min(i+1,n)/n)],fill='#34434c',width=2)
   self.d.polygon(triangle,fill='#34434c')
  self.svg.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in points)+'" fill="none" stroke="#34434c" stroke-width="2" marker-end="url(#arrow)"'+(' stroke-dasharray="7 5"' if dash else '')+'/>')
  if label:self.text(*at,label,19)
 def save(self,name):
  self.im.save(O/(name+'.png'),dpi=(220,220));(O/(name+'.svg')).write_text(''.join(self.svg)+'</svg>',encoding='utf-8')

d=Diagram(1280,1100,'图5-1  知识增强问答总体架构')
d.box(60,80,300,100,'前文诊断算法\n故障类别／部件判断')
d.box(480,80,300,100,'使用者提出\n诊断相关知识问题')
d.box(900,80,300,100,'资料范围与条件\n活动版本／显式筛选')
d.edge([(360,130),(480,130)],'任务衔接',(420,98),True);d.edge([(780,130),(900,130)])
d.text(630,240,'离线知识组织：资料处理后建立索引与来源关联',22)
d.box(60,290,300,120,'领域技术资料\n手册、规章及公开资料')
d.box(480,290,300,120,'解析与中文显示\n证据分块／来源定位')
d.box(900,270,300,160,'BM25文本索引\n中文E5向量索引\nNeo4j关系与来源')
d.edge([(360,350),(480,350)]);d.edge([(780,350),(900,350)])
d.text(630,485,'在线检索与回答：图谱按需参与，结果可回溯来源',22)
d.box(900,540,300,120,'三路候选发现\n文本／向量／图谱路径')
d.box(480,540,300,120,'因果意图与路径门控\n通过则启用图谱\n否则回退文本',size=22)
d.box(60,540,300,120,'加权RRF融合\n来源去重与预算筛选')
d.edge([(1050,430),(1050,540)]);d.edge([(1200,130),(1240,130),(1240,600),(1200,600)])
d.edge([(900,600),(780,600)]);d.edge([(480,600),(360,600)])
d.box(60,780,300,100,'证据片段与选择器\n大模型组织回答')
d.box(480,780,300,100,'引用及不确定性检查\n回答支持性检查')
d.box(900,780,300,100,'带引用的完整回答\n或部分事实与不足说明')
d.edge([(210,660),(210,780)]);d.edge([(360,830),(480,830)]);d.edge([(780,830),(900,830)])
d.box(900,970,300,80,'资料片段高亮\n支持片段齐全的路径展示',size=21)
d.edge([(1050,880),(1050,970)])
d.text(460,1010,'虚线表示任务衔接；不表示已自动确认故障根因。',21)
d.save('figure-5-1-architecture')

d=Diagram(1280,920,'图5-2  图谱关系与资料证据的来源关联示意')
d.box(80,110,240,100,'实体 u\n来源与角色标识')
d.box(520,110,240,100,'实体 v\n来源与角色标识')
d.box(960,110,240,100,'实体 w\n来源与角色标识')
d.edge([(320,160),(520,160)],'关系 r1',(420,125));d.edge([(760,160),(960,160)],'关系 r2',(860,125))
d.box(240,340,340,110,'支持片段 c1\n版本、引用与定位区间')
d.box(700,340,340,110,'支持片段 c2\n版本、引用与定位区间')
d.edge([(410,340),(410,180)],'来源对照',(345,270),True)
d.edge([(870,340),(870,180)],'来源对照',(950,270),True)
d.box(400,560,480,110,'候选证据集合 {c1, c2}\n附带关系顺序及来源标识')
d.edge([(410,450),(410,505),(530,505),(530,560)]);d.edge([(870,450),(870,505),(750,505),(750,560)])
d.box(400,750,480,100,'来源片段组织为回答上下文\n回答陈述绑定句级引用')
d.edge([(640,670),(640,750)])
d.text(640,890,'结构示意：路径可达不等于对具体故障根因的确认。',21)
d.save('figure-5-2-provenance')

d=Diagram(1280,1450,'图5-3  门控检索与证据约束回答流程')
for y,h,label in [(80,80,'输入问题及可选资料条件'),(230,90,'选择活动资料／检查适用范围'),(390,90,'BM25、向量与有向路径候选发现'),(550,90,'因果意图与完整路径来源校验')]:
 d.box(390,y,500,h,label)
for a,b in [(160,230),(320,390),(480,550)]:d.edge([(640,a),(640,b)])
d.box(80,730,430,90,'通过门控：三路加权RRF')
d.box(770,730,430,90,'未通过：回退双文本RRF')
d.edge([(500,640),(500,680),(295,680),(295,730)],'合格路径存在',(300,655))
d.edge([(780,640),(780,680),(985,680),(985,730)],'不适用或无合格路径',(1005,655))
d.box(390,910,500,90,'来源去重／配额与上下文预算筛选')
d.edge([(295,820),(295,865),(520,865),(520,910)])
d.edge([(985,820),(985,865),(760,865),(760,910)])
d.box(390,1060,500,90,'按证据生成陈述及引用选择器')
d.edge([(640,1000),(640,1060)])
d.box(390,1210,500,90,'引用、不确定性与支持性检查')
d.edge([(640,1150),(640,1210)])
d.box(930,1180,290,120,'未通过或无有效证据\n明确说明不足\n不展示未经核实结论',size=21)
d.edge([(890,1255),(930,1255)])
d.edge([(890,275),(1245,275),(1245,1240),(1220,1240)],'无匹配资料',(1110,245))
d.box(390,1360,500,65,'输出带引用回答或支持的部分事实',size=22)
d.edge([(640,1300),(640,1360)],'通过检查',(720,1330))
d.save('figure-5-3-online-flow')
shutil.copyfile(__file__,O/'draw_chapter5_methods.py')
manifest={'kind':'method diagrams, no quantitative result generation','font':FONT,'svg_editable_text_and_shapes':True,'png_dpi':220,'publisher':'school template pending','files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.iterdir() if p.is_file()}}
(O/'figure-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Three SVG and PNG diagrams saved:',O)
