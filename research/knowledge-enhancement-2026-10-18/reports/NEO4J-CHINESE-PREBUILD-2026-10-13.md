# 预建库、Neo4j 与中文展示交付

用户确认采用预先保存的知识库，新增资料再同步解析；页面简化；图谱使用 Neo4j；中文优先展示并保留原文。

## 实际部署

Neo4j Community 5.26.32 已通过官方 Maven Central 的 `org.neo4j:neo4j:5.26.32` 及其依赖安装并运行。本地 Java 薄启动器采用官方 DatabaseManagementServiceBuilder / BoltConnector 接口，使用真实 Neo4j 引擎和磁盘数据。发行 ZIP 下载站对当前地区返回403，因此改用官方公开 Maven 构件，未使用代理或模拟数据库。相关官方文档：[嵌入与启动](https://neo4j.com/docs/java-reference/current/java-embedded/setup/)、[JVM要求](https://neo4j.com/docs/operations-manual/current/installation/requirements/)。

Bolt 仅监听 `127.0.0.1:7687`，已初始化强密码；密码保存于本地忽略目录和隔离应用配置，不写入仓库。应用实际使用 `neo4j` 主图存储，不走 SQLite 图谱回退。SQLite 继续保存文档目录、版本、文本索引与会话。

数据目录 `.runtime/neo4j/data-home`；库文件 `.runtime/neo4j/lib`；便携 JDK 与 Maven 位于 `.runtime/neo4j/runtime`。启动器源码见 `scripts/neo4j-runtime`，启动命令：

```powershell
./scripts/start_local_neo4j.ps1
.venv/Scripts/python.exe -X utf8 scripts/run_upstream_application.py --profile-root .runtime/fusion/profile
```

Neo4j 的官方 ZIP 此环境不可下载。已核查可用的替代安装流程为：从 Azul CDN 获取 Zulu21.52.203 JDK21.0.12.1 Windows x64 ZIP；从 Maven Central 获取 Apache Maven3.9.11-bin.zip；解压到上述 runtime；设置当前进程 JAVA_HOME 指向便携JDK；执行 Maven 编译和 dependency:copy-dependencies。无需安装 Windows 系统服务或修改全局环境。

```powershell
$taskRoot = (Get-Location).Path
$env:JAVA_HOME = "$taskRoot/.runtime/neo4j/runtime/zulu21.52.203-ca-jdk21.0.12.1-win_x64"
& .runtime/neo4j/runtime/apache-maven-3.9.11/bin/mvn.cmd -B -f scripts/neo4j-runtime/pom.xml "-Dmaven.repo.local=$taskRoot/.runtime/neo4j/maven-repository" package dependency:copy-dependencies
```

启动器配置项目内 Java socket 临时目录，修复此 Windows 环境默认临时路径上的 Unix-domain 回环错误，使用JDK受支持的回环路径；认证保持启用。当前部署提供 Bolt，不声称安装了 Neo4j Browser 管理网站。

## 预建与增量处理

初始资料来自4份已校验原文件：FAA AMT Powerplant第5章、第10章，FAA AC33.63-1，CAAC CCAR-33-R2。按压气机污染/损伤、燃烧室、涡轮检查、起动性能、适航稳定性/结冰/响应及振动指导组织为10份来源节选，共39页、96文本块；另保留原FAA单页资料，活动资料总数11份。

这是初始工程资料库，节选不代表原文件完整内容或主题全面覆盖。节选PDF名称明确标记“资料节选”，原文件SHA256、URL、原物理页码与派生页码映射保存在 `.runtime/fusion/source-catalog`。没有导入题目或标准答案。通用训练与规章资料不自动视为具体型号的现行维修指令。

预处理目录管理脚本 `scripts/prebuild_knowledge.py` 检查资料哈希、元数据、模型、切分预算与tokenizer变化，复用活动版本。实际再次运行10份全部 reused；未改变的内容不重复抽取。新上传或更新资料由现有应用导入链路解析、索引、抽取、翻译和写入Neo4j，完成后发布新版本。历史版本保留，活动检索只读最新已完成版本。

```powershell
.venv/Scripts/python.exe -X utf8 scripts/prebuild_knowledge.py --catalog .runtime/fusion/source-catalog/catalog.json --profile-root .runtime/fusion/profile
```

旧 SQLite 图谱通过 `scripts/migrate_graph_to_neo4j.py` 实际迁移，逐实体/关系比较ID及全部属性，迁移校验通过。切换后又经 `/api/documents/upload` 更新FAA单页来源信息，验证新版本直接写Neo4j，旧版本superseded且不进入活动图谱。

## 中文展示与页面

只维护一套实体、关系和ID。原名称、原说明和精确引用不改变，中文译名/说明保存在 properties 的 name_zh / description_zh 中，并保存机器翻译标记及源字符串哈希。中文与原文切换只改变显示，关系代码和证据定位保持原样。Neo4j中文搜索字段同步更新；当前144个活动节点全部具备中文展示，72条活动关系。

英文证据保留英文，可切换原文或下载来源核查；不把机器译文当成原始引文。首页“知识适用条件”折叠为高级筛选，知识管理来源信息也默认折叠。元数据和条件冲突检查继续保留。侧栏窄窗口布局已调整并进行浏览器验收。

## 验收

全量483项pytest通过，Ruff通过，Mypy98源文件通过；最后的前端中文/原文与类型显示检查通过。真实Neo4j中文问答产生 evidence_ready、3条核验陈述，引用同时来自文档块和图谱路径。冻结正式实验成绩未随本次部署变更。完整机器收据见 [验收JSON](neo4j-chinese-prebuild-verification.json)。

所有模型调用继续复用 `.runtime/thesis-research-provider-v1` 和原5元pilot上限，不重置预算。实际累计费用及调用数以验收JSON为准。

本轮未提交、合并或发布。

## 用户要求更新

本记录中的“中文/原文切换”已按用户最新意见取消。现在业务界面为中文，图谱全部同页显示并支持类型按钮隐藏/恢复，具体见 [中文图谱优化记录](GRAPH-CHINESE-UX-2026-10-13.md)。
