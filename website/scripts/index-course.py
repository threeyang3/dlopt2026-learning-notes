"""Read-only extraction: the website reuses original notes, never rewrites them."""
import json,re
from pathlib import Path
root=Path(__file__).resolve().parents[2]
prereqs=[[],[1],[2],[1],[2,3],[4,5],[1,3],[7],[6,8],[3,9],[4,5,10],[6],[8],[7,8,13]]
labs=['attention','linear','linear','sgd','gd','sgd','attention','flash','resources','subspace','zeroth','distributed','precision','inference']
items=[]
for i,p in enumerate(sorted((root/'notes').glob('Lecture-*.md'))):
 text=p.read_text()
 goals=re.search(r'## 学习目标\s+(.+?)(?=\n## )',text,re.S).group(1).strip()
 pdfs=sorted((root/'materials'/p.stem).glob('*.pdf'))
 items.append(dict(id=i+1,slug=p.stem,title=text.splitlines()[0].removeprefix('# '),english=p.stem.split('_',1)[1].replace('-',' '),goals=goals,prerequisites=prereqs[i],topics=re.findall(r'^## \d+\. (.+)',text,re.M),note='notes/'+p.name,pdfs=[str(x.relative_to(root)) for x in pdfs],lab=labs[i],status='partially_verified',evidence='原笔记与 SOURCE_AUDIT.md 已对照；网站对应计算引擎另有数值测试。整讲及所有 PDF 未在本轮独立逐式复核。',category='课程核心',scope='Lite / MuonM 的定义仍需课堂材料确认；Sophia 等见笔记扩展。' if i==8 else '历史年份资料若被当年官网列出，仍属课程核心。',sourceLocations=text.split('## 资料定位')[-1].split('\n下一')[0].split('\n回到')[0].strip()))
(root/'website/src/content/course.json').write_text(json.dumps(items,ensure_ascii=False,indent=2)+'\n')
