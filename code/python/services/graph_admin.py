"""
图谱管理服务 — 软删除 + 详情查询 + 运维审计

职责:
  1. 实体软删除（设置 is_deleted=true，不真正删除）
  2. 关系软删除
  3. 实体详情三层查询（业务属性 / 元数据 / 邻接节点）
  4. 操作日志记录
"""

from __future__ import annotations

import time
from typing import Any

from config import settings
from utils.log import logger


class GraphAdminService:
    """图谱运维管理服务"""

    def __init__(self, knowledge_graph: Any = None) -> None:
        self.kg = knowledge_graph
        self._op_log: list[dict] = []

    async def delete_entity(self, name: str) -> dict:
        """软删除实体 — 设置 is_deleted=true"""
        cypher = """
        MATCH (e:Entity {name: $name})
        SET e.is_deleted = true, e.deleted_at = $now
        RETURN e.name AS name, e.type AS type
        """
        try:
            records = await self.kg.execute_cypher(cypher, {"name": name, "now": int(time.time())})
            self._log_op("delete_entity", name, success=True)
            if records:
                return {"status": "deleted", "entity": records[0]}
            return {"status": "not_found", "entity": None}
        except Exception as e:
            self._log_op("delete_entity", name, success=False, error=str(e))
            logger.error(f"GraphAdmin delete_entity failed: {e}")
            return {"status": "error", "message": str(e)}

    async def delete_relation(self, head: str, rel: str, tail: str) -> dict:
        """软删除关系 — 参数化 Cypher 防止注入"""
        import re
        # Validate relation type is safe (alphanumeric + underscore only)
        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', rel):
            logger.error(f"GraphAdmin delete_relation rejected unsafe relation type: {rel}")
            return {"status": "error", "message": f"Invalid relation type: {rel}"}
        cypher = f"""
        MATCH (h:Entity {{name: $head}})-[r:{rel}]->(t:Entity {{name: $tail}})
        SET r.is_deleted = true, r.deleted_at = $now
        RETURN h.name AS head, type(r) AS relation, t.name AS tail
        """
        try:
            records = await self.kg.execute_cypher(cypher, {"head": head, "tail": tail, "now": int(time.time())})
            self._log_op("delete_relation", f"{head}-[{rel}]->{tail}", success=True)
            if records:
                return {"status": "deleted", "relation": records[0]}
            return {"status": "not_found", "relation": None}
        except Exception as e:
            self._log_op("delete_relation", f"{head}-[{rel}]->{tail}", success=False, error=str(e))
            logger.error(f"GraphAdmin delete_relation failed: {e}")
            return {"status": "error", "message": str(e)}

    async def get_entity_detail(self, name: str) -> dict:
        """获取实体三层详情

        Returns:
            {"entity": {...}, "metadata": {...}, "neighbors": [...]}
        """
        # Layer 1: 业务属性
        entity_cypher = "MATCH (e:Entity {name: $name}) WHERE e.is_deleted IS NULL OR e.is_deleted = false RETURN e"
        entity_records = await self.kg.execute_cypher(entity_cypher, {"name": name})
        entity_data = entity_records[0] if entity_records else None

        if not entity_data:
            return {"entity": None, "metadata": None, "neighbors": []}

        # Layer 2: 元数据（版本、时间戳等）
        metadata = {}
        e_node = entity_data.get("e", {})
        if hasattr(e_node, "items"):
            metadata = {
                "version": e_node.get("version", ""),
                "created_at": e_node.get("created_at", ""),
                "updated_at": e_node.get("updated_at", ""),
                "source": e_node.get("source", ""),
            }

        # Layer 3: 邻接节点
        neighbor_cypher = """
        MATCH (e:Entity {name: $name})-[r]-(n:Entity)
        WHERE (e.is_deleted IS NULL OR e.is_deleted = false)
          AND (n.is_deleted IS NULL OR n.is_deleted = false)
          AND (r.is_deleted IS NULL OR r.is_deleted = false)
        RETURN n.name AS name, n.type AS type, type(r) AS relation, n.description AS description
        LIMIT 50
        """
        neighbors = await self.kg.execute_cypher(neighbor_cypher, {"name": name})

        return {"entity": entity_data, "metadata": metadata, "neighbors": neighbors}

    async def get_logs(self, limit: int = 50) -> list[dict]:
        """获取操作审计日志"""
        return self._op_log[-limit:]

    async def get_graph_for_viz(self, limit: int = 100) -> dict:
        """获取全图谱数据（过滤已删除），供前端 ECharts 可视化"""
        cypher = """
        MATCH (n:Entity)-[r]->(m:Entity)
        WHERE (n.is_deleted IS NULL OR n.is_deleted = false)
          AND (m.is_deleted IS NULL OR m.is_deleted = false)
          AND (r.is_deleted IS NULL OR r.is_deleted = false)
        RETURN n.name AS source_name, n.type AS source_type, n.description AS source_desc,
               type(r) AS relation,
               m.name AS target_name, m.type AS target_type, m.description AS target_desc
        LIMIT $limit
        """
        records = await self.kg.execute_cypher(cypher, {"limit": limit})

        nodes: dict[str, dict] = {}
        edges: list[dict] = []
        for rec in records:
            src = rec.get("source_name", "")
            tgt = rec.get("target_name", "")
            if src not in nodes:
                nodes[src] = {"name": src, "type": rec.get("source_type", ""), "description": rec.get("source_desc", "")}
            if tgt not in nodes:
                nodes[tgt] = {"name": tgt, "type": rec.get("target_type", ""), "description": rec.get("target_desc", "")}
            edges.append({
                "source": src,
                "target": tgt,
                "relation": rec.get("relation", ""),
            })
        return {"nodes": list(nodes.values()), "edges": edges}

    def _log_op(self, action: str, target: str, success: bool, error: str = "") -> None:
        self._op_log.append({
            "action": action,
            "target": target,
            "success": success,
            "error": error,
            "time": int(time.time()),
        })
        if len(self._op_log) > 500:
            self._op_log = self._op_log[-500:]
