# AeroDiagnosis 自顶向下应用框架

## 目标

新实现围绕可验证业务用例组织，而不是围绕 Web 框架、Agent 类或某个数据库组织。旧
`code/python` 仅作为迁移期行为参照；任何新入口都必须调用 `src/aerodiagnosis` 中的应用
用例，不允许复制业务流程。

## 依赖方向

```text
FastAPI / CLI / future MCP
            |
            v
    application use cases
        |             |
        v             v
  domain model       ports
                       ^
                       |
              persistence adapters
```

依赖只能向内：

1. `domain` 不依赖 FastAPI、数据库、LangGraph 或模型 SDK；
2. `application` 编排用例，只依赖领域模型与端口；
3. `adapters` 把 HTTP、CLI、SQLite 等技术协议映射到端口或用例；
4. `bootstrap.py` 是唯一知道具体适配器的组合根；
5. API/MCP 不得各自实现一套摄取、检索或诊断语义。

## 当前纵向切片

```text
document bytes
  -> verified parser adapter
  -> immutable version + stable chunk/evidence IDs
  -> staged vector records
  -> manifest atomic active pointer
  -> active-version-filtered evidence search
```

SQLite 清单是可见性的唯一权威。向量记录写入后，只有清单成功切换为 `active` 才会出现在
检索结果中；历史记录和失败记录不会被误当作当前证据。当前实现保留历史以便审计，清理和
重放由后续持久 worker 负责。

## 下一层实现顺序

1. 已完成四个只读领域工具：手册检索、图谱遍历、案例检索、参数分析；
2. 已完成模型驱动显式状态机、SQLite checkpoint、持久短期记忆和 fail-closed verifier；
3. 补齐模型调用 operation ledger、deadline、token/cost 预算和人工复核节点；
4. FastAPI 与 MCP 映射同一诊断用例和错误注册表；
5. 用冻结评测集决定是否迁移 LangGraph，以及检索融合、纠错和路由策略。
