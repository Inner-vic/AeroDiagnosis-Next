# Public Aero Corpus v1.1 数据卡

## 用途

这是面向 AeroDiagnosis 研究管道的小规模公开语料资产，用于：

- 验证 PDF 与官方全文文本的解析差异；
- 形成航空发动机气路诊断知识抽取候选；
- 构建单路、多路与排序算法的检索评测问题池；
- 演示来源、许可、哈希和派生数据的可追溯管理。

它不是领域专家金标准，也不能用于证明诊断准确率或维修有效性。

## 数据组成

- 10 份公开文档或数据源：NASA 技术文献、FAA 维护手册、NTSB 调查报告、Federal Register 适航指令、Europe PMC 开放论文和 FAA SDR 故障记录；
- 17 个原始文件，覆盖 PDF、TXT、HTML、XML、JSON 和 CSV 六类媒体格式；
- 536 个段落或记录感知文本块；
- 24 条带来源定位的知识抽取候选；
- 30 个检索查询与文档级分级相关性初标；
- 16 个解析评测样本定义。

原始文件和派生全文位于 `.runtime/datasets/public-aero-corpus-v1/`，受 `.gitignore` 保护。仓库只保存可复现构建所需的来源清单、哈希、标注初稿与基线报告。

## 构建

```powershell
.\scripts\Build-PublicCorpus.ps1
```

离线验证已经下载的文件：

```powershell
.\scripts\Build-PublicCorpus.ps1 -Offline
```

构建器会严格核验每个文件的字节数和 SHA-256，并执行：

- 提取 NASA、FAA 和 NTSB PDF；允许读取具有空口令权限保护的公开 PDF，但拒绝需要密码的文档；
- 解析 JATS XML 的标题、摘要、正文、表格和图注；
- 比较同一适航指令的 HTML、XML 与纯文本包装页，记录跨格式 token overlap；
- 从 66,079 条 FAA 2024 SDR 中筛出 1,997 条发动机系统记录，再按 JASC 71–79 每类确定性抽取 25 条；
- 不输出运营人、提交者、注册号和各类序列号字段，仅保留研究所需的设备、部件、状态与故障叙述。

生成文件包括：

- `processed/*.txt`：统一 Unicode 与换行后的文本；
- `processed/corpus.jsonl`：确定性段落块、字符位置和块哈希；
- `processed/faa-sdr-2024-engine-sample.csv`：225 条去除直接标识字段的发动机系统分层样本；
- `processed/build_report.json`：页数、字符数、块数和解析对照指标。

## 当前解析基线

NASA 官方全文是另一条机器解析结果，不是人工金标。因此当前 token overlap 只能用于发现明显解析退化，不能称为 OCR 或版面恢复准确率。四份 NASA 文献的 token F1 为 0.9454–0.9977。

Federal Register 同一适航指令的 HTML→XML 与文本包装页→XML token F1 分别为 0.9848 和 0.9861。这仍是机器格式之间的一致性信号，不是结构、表格或语义正确率。

FAA 页 60 包含“故障现象—可能原因—建议动作”三列表格，页 64 包含双栏正文与图片；NTSB 第 1–2 页包含报告头、根因叙述和 findings 表；两篇 JATS XML 共包含 17 张表。这些复杂样本后续需要人工制作结构和阅读顺序金标。

## 标注等级

- `machine_candidate`：由公开资料和自动过程形成，只能作为待审候选；
- `silver`：两名标注者一致或完成分歧仲裁；
- `gold`：具备相关背景的复核者确认，并冻结版本与哈希。

当前知识抽取与检索标注均为 `machine_candidate`，不得在论文中称为金标准。

## 许可与限制

NASA NTRS 元数据将四份文献标记为 `PUBLIC`，版权判定为 `PUBLIC_USE_PERMITTED` 或 `GOV_PUBLIC_USE_PERMITTED`，且出口控制为 `NO`。NTSB 调查报告、FAA SDR 和 Federal Register 适航指令均为美国联邦政府公开材料；Federal Register HTML/XML 是信息呈现版本，法律研究仍应核对官方版本。

两篇 Europe PMC JATS 论文采用 CC BY 4.0，派生文本须保留署名和来源。C-MAPSS 数值数据页面虽然标记公开访问，但同时显示未指定许可证，因此本版本只记录该来源，没有下载或再分发其载荷。

FAA 文档为官方公开出版物，FAA 的复制/修改责任声明适用。任何派生文本都必须保留来源，不得表示为 FAA 审核或认可的修改版本。

质量筛选排除了两篇候选论文：PMC8464420 已被撤稿；PMC10468384 为 CC BY-NC-ND 4.0，为避免派生文本的改编权边界不清而不纳入。

## 已知偏差

- 全部语料仍为英文，政府机构材料占比较高；
- 诊断基准、科研论文、维护手册、监管规则、事故调查和自愿故障报告的写作目的不同；
- 文档级 qrels 粒度较粗，尚未完成块级穷尽判断；
- 领域覆盖集中在气路诊断、C-MAPSS 和典型涡轮发动机排查；
- FAA SDR 是按 JASC 大类均衡抽取的研究样本，不代表真实故障类别分布；自由文本未经事实核验；
- 自动候选可能包含概念粒度不一致或关系方向错误。

正式实验前必须扩展中文语料、增加负例与同义查询，并完成双人独立标注和专家抽样复核。
