"""封签库存分配规划。

处理流程：
1. 对请求中的 ``chains`` **复用批量封签占用检查**（``validate_batch``），原样保留
   批量校验结果；其中任何链的**有效** seal/reseal 事件（status=ok）占用的 seal_id
   都从分配池中排除（含跨链冲突编号——冲突只在批量结果中标记，不改变“该编号已被
   有效占用”的事实）；
2. 对每条待用记录与每枚剩余库存封签建立可分配边：``use_at`` 严格早于
   ``expires_at``（到期时间不含）且记录保管人在该封签允许保管人名单中；
3. 在全部可分配边上做**一对一全局最优分配**（最小费用最大流，绝非逐条贪心），
   目标依次为：
     a. 最大化满足（assigned）记录数；
     b. 满足记录数相同时，最小化全部已分配边的到期余量总秒数
        （``expires_at - use_at``）；
     c. 仍并列时，按待用记录输入顺序逐条比较、封签取 seal_id 字典序最小者，
        确定唯一方案（具体比较元：按记录顺序排列的“分配封签字典序名次 /
        未分配”序列，字典序最小）；
4. 逐条返回 assigned / unassigned、分配编号与未分配原因：
   - no_candidate：排除占用后不存在任何满足“未过期 + 保管人允许”的候选封签；
   - competition：存在候选，但在最大化满足数的一对一全局分配中候选封签全部分给
     其他记录而被竞争淘汰，响应逐候选给出最终分得该封签的记录下标。
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from .batch import validate_batch
from .models import (
    BatchValidateRequest,
    ChainEvents,
    CompetitionWinner,
    CrossChainConflict,
    EventType,
    ExcludedInventorySeal,
    InventorySeal,
    PendingSealUse,
    SealAllocationResult,
    SealInventoryPlanRequest,
    SealInventoryPlanResponse,
    SealInventoryPlanSummary,
)

_SEAL_EVENTS = (EventType.SEAL, EventType.RESEAL)

# 单条边到期余量的安全上限（微秒），用于构造严格分层的整数费用；
# 现实时间差不可能超过该值，仅用于数值上保证字典序层严格压过余量层。
_SLACK_CAP_MICROSECONDS = 10**20


class SealInventoryPlanError(ValueError):
    """分配规划入参的语义错误（在路由层转换为 HTTP 422）。"""


# ---------------------------------------------------------------------------
# 最小费用流（整数费用，势能 + Dijkstra）
# ---------------------------------------------------------------------------


@dataclass
class _Edge:
    """残量网络中的一条有向边。"""

    to: int
    rev: int
    cap: int
    cost: int


class _MinCostFlow:
    """整数费用最小费用流；初始费用非负，初始势能取零，后续以最短路距离更新。"""

    def __init__(self, n: int) -> None:
        self.n = n
        self.graph: List[List[_Edge]] = [[] for _ in range(n)]

    def add_edge(self, fr: int, to: int, cap: int, cost: int) -> _Edge:
        forward = _Edge(to=to, rev=len(self.graph[to]), cap=cap, cost=cost)
        backward = _Edge(to=fr, rev=len(self.graph[fr]), cap=0, cost=-cost)
        self.graph[fr].append(forward)
        self.graph[to].append(backward)
        return forward

    def min_cost_flow(self, source: int, sink: int, required: int) -> int:
        """推送恰好 required 单位流，返回最小总费用；本问题中必可行。"""
        n = self.n
        potential = [0] * n
        total_cost = 0
        flow = 0
        while flow < required:
            dist: List[Optional[int]] = [None] * n
            prev_edge: List[Optional[_Edge]] = [None] * n
            prev_vertex: List[int] = [-1] * n
            dist[source] = 0
            heap: List[Tuple[int, int]] = [(0, source)]
            while heap:
                d, v = heapq.heappop(heap)
                if d != dist[v]:
                    continue
                for edge in self.graph[v]:
                    if edge.cap <= 0:
                        continue
                    reduced = edge.cost + potential[v] - potential[edge.to]
                    nd = d + reduced
                    if dist[edge.to] is None or nd < dist[edge.to]:  # type: ignore[operator]
                        dist[edge.to] = nd
                        prev_vertex[edge.to] = v
                        prev_edge[edge.to] = edge
                        heapq.heappush(heap, (nd, edge.to))

            if dist[sink] is None:  # pragma: no cover - 记录直连汇点，必可行
                raise RuntimeError("分配网络不存在可行流")
            for v in range(n):
                if dist[v] is not None:
                    potential[v] += dist[v]  # type: ignore[operator]

            # 每条边容量均为 1，每次增广 1 单位
            v = sink
            while v != source:
                edge = prev_edge[v]
                assert edge is not None
                edge.cap -= 1
                self.graph[v][edge.rev].cap += 1
                v = prev_vertex[v]
            total_cost += potential[sink]
            flow += 1
        return total_cost


def _ensure_comparable(
    chains: Sequence[ChainEvents],
    records: Sequence[PendingSealUse],
    seals: Sequence[InventorySeal],
) -> None:
    """事件、待用时间与到期时间混用带时区与不带时区时间时统一报 422。"""
    timestamps: List[datetime] = []
    for chain in chains:
        timestamps.extend(event.timestamp for event in chain.events)
    timestamps.extend(record.use_at for record in records)
    timestamps.extend(seal.expires_at for seal in seals)
    if not timestamps:
        return
    aware = timestamps[0].tzinfo is not None and timestamps[0].utcoffset() is not None
    for value in timestamps[1:]:
        value_aware = value.tzinfo is not None and value.utcoffset() is not None
        if value_aware != aware:
            raise SealInventoryPlanError(
                "时间戳时区信息不一致（不能混用带时区与不带时区的时间）"
            )


def _slack_microseconds(record: PendingSealUse, seal: InventorySeal) -> int:
    """到期余量（微秒）；调用方已保证 use_at < expires_at。"""
    delta = seal.expires_at - record.use_at
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def plan_seal_inventory(request: SealInventoryPlanRequest) -> SealInventoryPlanResponse:
    """复用批量校验排除占用编号，再在可行边上求一对一全局最优分配。"""
    _ensure_comparable(request.chains, request.records, request.seals)

    chain_ids = {chain.chain_id for chain in request.chains}
    for i, record in enumerate(request.records):
        if record.chain_id not in chain_ids:
            raise SealInventoryPlanError(
                f"第 {i} 条待用记录引用的 chain_id 不存在：{record.chain_id}"
            )

    # 阶段 1：复用批量校验（其内部先逐链状态机校验，再建立全批次占用表）
    batch_response = validate_batch(
        BatchValidateRequest(chains=list(request.chains))
    )

    # 收集每个 seal_id 的全部有效 seal/reseal 占用（按链输入顺序）
    occupations: Dict[str, List[CrossChainConflict]] = {}
    for chain, chain_result in zip(request.chains, batch_response.chains):
        for event, event_result in zip(chain.events, chain_result.events):
            if event_result.status != "ok" or event.type not in _SEAL_EVENTS:
                continue
            seal_id = event.seal_id
            if seal_id is None:  # pragma: no cover - 有效事件必带 seal_id
                continue
            prior = occupations.setdefault(seal_id, [])
            first_chain_id = prior[0].chain_id if prior else chain.chain_id
            prior.append(
                CrossChainConflict(
                    seal_id=seal_id,
                    chain_id=chain.chain_id,
                    event_index=event_result.index,
                    first_chain_id=first_chain_id,
                )
            )

    # 阶段 2：排除已占用编号，剩余库存按 seal_id 字典序确定并列方案
    excluded: List[ExcludedInventorySeal] = []
    pool: List[InventorySeal] = []
    for seal in request.seals:
        if seal.seal_id in occupations:
            excluded.append(
                ExcludedInventorySeal(
                    seal_id=seal.seal_id,
                    occupied_by=list(occupations[seal.seal_id]),
                )
            )
        else:
            pool.append(seal)
    excluded.sort(key=lambda item: item.seal_id)
    pool.sort(key=lambda item: item.seal_id)

    n, m = len(request.records), len(pool)

    # 每条记录的候选（可行边）封签下标，封签已按 seal_id 字典序排列，
    # 故候选列表本身即按 seal_id 字典序。
    candidates: List[List[int]] = [[] for _ in range(n)]
    slacks: Dict[Tuple[int, int], int] = {}
    for i, record in enumerate(request.records):
        for j, seal in enumerate(pool):
            if (
                record.custodian in seal.allowed_custodians
                and record.use_at < seal.expires_at
            ):
                candidates[i].append(j)
                slacks[(i, j)] = _slack_microseconds(record, seal)

    # 阶段 3：最小费用流求一对一全局最优。
    # 节点布局：源点 0；记录 1..n；封签 1+n..n+m；汇点 n+m+1。
    source, sink = 0, n + m + 1
    mcf = _MinCostFlow(sink + 1)
    record_nodes = [1 + i for i in range(n)]
    seal_nodes = [1 + n + j for j in range(m)]

    # 整数费用严格分层（优先级：满足数 > 总余量 > 字典序）：
    #   lex_value  = 按记录输入顺序排列的位次数字序列（匹配第 j 个封签记 j 即
    #                seal_id 字典序名次，未分配记 m=最大名次+1）的 B 进制值，
    #                数字越小越优——越早位置的记录越优先被分配、拿到字典序越小的封签；
    #   W_lex      = B**n，严格大于 lex_value 的最大可能取值（B**n - 1）；
    #   W_count    再严格压过总余量（微秒）层的任意差，使少满足一条记录必为更劣。
    # 边费用 = 未分配惩罚 W_count（仅未分配边）+ 该边余量 * W_lex + 该位次数字权值。
    base = m + 1
    lex_max = base**n - 1
    w_lex = lex_max + 1
    w_count = w_lex * (n * _SLACK_CAP_MICROSECONDS + 1)

    match_edges: Dict[Tuple[int, int], _Edge] = {}
    for i, record_node in enumerate(record_nodes):
        mcf.add_edge(source, record_node, 1, 0)
        digit_weight = base ** (n - 1 - i)
        # 未分配边：满足数惩罚 + 位次数字 m（大于一切匹配名次）
        mcf.add_edge(record_node, sink, 1, w_count + m * digit_weight)
        for j in candidates[i]:
            edge = mcf.add_edge(
                record_node,
                seal_nodes[j],
                1,
                slacks[(i, j)] * w_lex + j * digit_weight,
            )
            match_edges[(i, j)] = edge
    for seal_node in seal_nodes:
        mcf.add_edge(seal_node, sink, 1, 0)

    # 全部记录必须各走 1 单位流（经封签或直连汇点的未分配边）
    mcf.min_cost_flow(source, sink, n)

    assigned_seal: Dict[int, int] = {}
    seal_owner: Dict[int, int] = {}
    for (i, j), edge in match_edges.items():
        if edge.cap == 0:  # 正向容量耗尽即该边输送了 1 单位流
            assigned_seal[i] = j
            seal_owner[j] = i

    # 阶段 4：逐条结果
    occupied_inventory_count = len(excluded)
    results: List[SealAllocationResult] = []
    no_candidate_count = competition_count = 0
    total_slack_micro = 0
    for i, record in enumerate(request.records):
        candidate_ids = [pool[j].seal_id for j in candidates[i]]
        if i in assigned_seal:
            j = assigned_seal[i]
            slack = slacks[(i, j)]
            total_slack_micro += slack
            results.append(
                SealAllocationResult(
                    index=i,
                    chain_id=record.chain_id,
                    custodian=record.custodian,
                    use_at=record.use_at,
                    status="assigned",
                    seal_id=pool[j].seal_id,
                    slack_seconds=slack / 1_000_000,
                )
            )
            continue

        if not candidates[i]:
            no_candidate_count += 1
            expired = sum(
                1 for seal in pool if record.use_at >= seal.expires_at
            )
            forbidden = sum(
                1
                for seal in pool
                if record.custodian not in seal.allowed_custodians
            )
            reason = (
                "无候选封签：库存中不存在同时满足未被有效 seal/reseal 占用、"
                "use_at 早于 expires_at、保管人在允许名单内的封签"
                f"（库存 {len(request.seals)} 枚，已占用 {occupied_inventory_count} 枚，"
                f"剩余中已过期 {expired} 枚、保管人不允许 {forbidden} 枚，两类可重叠）"
            )
            results.append(
                SealAllocationResult(
                    index=i,
                    chain_id=record.chain_id,
                    custodian=record.custodian,
                    use_at=record.use_at,
                    status="unassigned",
                    reason_code="no_candidate",
                    reason=reason,
                )
            )
            continue

        # 最大基数匹配下，该记录的每个候选封签必然都分给了其他记录，
        # 否则可直接增广一条边而提高满足数。
        competition_count += 1
        competition = [
            CompetitionWinner(
                seal_id=pool[j].seal_id,
                assigned_record_index=seal_owner[j],
            )
            for j in candidates[i]
        ]
        reason = (
            f"存在 {len(candidates[i])} 个候选封签，但在最大化满足记录数的"
            "一对一全局分配中全部被其他记录取得而淘汰（非逐条贪心结果）"
        )
        results.append(
            SealAllocationResult(
                index=i,
                chain_id=record.chain_id,
                custodian=record.custodian,
                use_at=record.use_at,
                status="unassigned",
                reason_code="competition",
                reason=reason,
                candidate_seal_ids=candidate_ids,
                competition=competition,
            )
        )

    summary = SealInventoryPlanSummary(
        assigned=len(assigned_seal),
        unassigned=n - len(assigned_seal),
        no_candidate=no_candidate_count,
        competition=competition_count,
        total_slack_seconds=total_slack_micro / 1_000_000,
        available_seals=m,
    )

    return SealInventoryPlanResponse(
        batch_validation=batch_response,
        results=results,
        excluded_inventory=excluded,
        summary=summary,
    )
