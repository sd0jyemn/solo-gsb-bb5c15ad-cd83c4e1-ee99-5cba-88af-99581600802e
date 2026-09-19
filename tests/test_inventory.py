"""封签库存分配规划的单元测试与 API 集成测试。"""
from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.inventory import SealInventoryPlanError, plan_seal_inventory
from app.main import app
from app.models import (
    ChainEvents,
    Event,
    EventType,
    InventorySeal,
    PendingSealUse,
    SealInventoryPlanRequest,
)

client = TestClient(app)

DAY = "2026-09-19"


def dt(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime.fromisoformat(f"{DAY}T{hour:02d}:{minute:02d}:{second:02d}+00:00")


def ev(type_: EventType, hour: int, minute: int = 0, **kw) -> Event:
    return Event(type=type_, timestamp=dt(hour, minute), **kw)


def chain(chain_id: str, events: list[Event] | None = None) -> ChainEvents:
    return ChainEvents(chain_id=chain_id, events=events or [])


def record(
    custodian: str = "Alice",
    use_at: datetime | None = None,
    chain_id: str = "c1",
) -> PendingSealUse:
    return PendingSealUse(chain_id=chain_id, custodian=custodian, use_at=use_at or dt(10))


def seal(
    seal_id: str,
    expires_hour: int,
    custodians=("Alice",),
    expires_minute: int = 0,
) -> InventorySeal:
    return InventorySeal(
        seal_id=seal_id,
        expires_at=dt(expires_hour, expires_minute),
        allowed_custodians=list(custodians),
    )


def make_request(chains=(), records=(), seals=()) -> SealInventoryPlanRequest:
    return SealInventoryPlanRequest(
        chains=list(chains), records=list(records), seals=list(seals)
    )


def by_index(resp):
    return {item.index: item for item in resp.results}


# ---------------------------------------------------------------------------
# 基本分配：可行边判定（未过期 + 保管人允许，到期时间不含）
# ---------------------------------------------------------------------------


def test_simple_assignment_returns_slack():
    req = make_request(
        chains=[chain("c1")],
        records=[record("Alice", dt(10))],
        seals=[seal("S1", 12)],
    )
    resp = plan_seal_inventory(req)

    item = resp.results[0]
    assert item.status == "assigned"
    assert item.seal_id == "S1"
    assert item.slack_seconds == 7200.0
    assert item.reason_code is None and item.reason is None
    assert item.candidate_seal_ids == [] and item.competition == []
    assert resp.excluded_inventory == []
    assert resp.summary.model_dump() == {
        "assigned": 1,
        "unassigned": 0,
        "no_candidate": 0,
        "competition": 0,
        "total_slack_seconds": 7200.0,
        "available_seals": 1,
    }
    # 原批量校验结果原样保留（空链合法）
    assert resp.batch_validation.valid is True
    assert resp.batch_validation.chains[0].chain_id == "c1"


def test_expiry_is_half_open_use_at_equals_expires_at_has_no_edge():
    req = make_request(
        chains=[chain("c1")],
        records=[record("Alice", dt(12))],
        seals=[seal("S1", 12)],
    )
    resp = plan_seal_inventory(req)
    item = resp.results[0]
    assert item.status == "unassigned"
    assert item.reason_code == "no_candidate"
    assert item.seal_id is None and item.slack_seconds is None


def test_custodian_not_allowed_has_no_edge():
    req = make_request(
        chains=[chain("c1")],
        records=[record("Bob", dt(10))],
        seals=[seal("S1", 12, custodians=("Alice",))],
    )
    resp = plan_seal_inventory(req)
    assert resp.results[0].reason_code == "no_candidate"
    assert resp.summary.no_candidate == 1


def test_no_candidate_reason_lists_counts():
    req = make_request(
        chains=[chain("c1")],
        records=[record("Bob", dt(11))],
        seals=[
            seal("S-EXPIRED", 10, custodians=("Alice", "Bob")),  # 已过期但保管人允许
            seal("S-FORBIDDEN", 12, custodians=("Alice",)),
        ],
    )
    resp = plan_seal_inventory(req)
    assert "已过期 1 枚" in resp.results[0].reason
    assert "保管人不允许 1 枚" in resp.results[0].reason


def test_empty_request_succeeds():
    resp = plan_seal_inventory(make_request())
    assert resp.results == []
    assert resp.excluded_inventory == []
    assert resp.summary.assigned == 0
    assert resp.summary.total_slack_seconds == 0.0
    assert resp.batch_validation.valid is True


# ---------------------------------------------------------------------------
# 阶段 1：复用批量校验排除已占用编号
# ---------------------------------------------------------------------------


def test_seal_used_by_valid_seal_event_is_excluded():
    chains = [
        chain("c1", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")]),
    ]
    req = make_request(
        chains=chains,
        records=[record("Alice", dt(10), chain_id="c1")],
        seals=[seal("S1", 12)],
    )
    resp = plan_seal_inventory(req)

    assert [item.seal_id for item in resp.excluded_inventory] == ["S1"]
    occupied = resp.excluded_inventory[0].occupied_by
    assert len(occupied) == 1
    assert occupied[0].chain_id == "c1"
    assert occupied[0].first_chain_id == "c1"
    assert occupied[0].event_index == 0
    assert resp.results[0].reason_code == "no_candidate"
    assert resp.summary.available_seals == 0


def test_seal_used_by_valid_reseal_event_is_excluded():
    chains = [
        chain("c1", [
            ev(EventType.SEAL, 8, seal_id="S0", custodian="Alice"),
            ev(EventType.UNSEAL, 9),
            ev(EventType.RESEAL, 9, 30, seal_id="S9", custodian="Alice"),
        ]),
    ]
    req = make_request(
        chains=chains,
        records=[record("Alice", dt(10), chain_id="c1")],
        seals=[seal("S9", 12), seal("S7", 13)],
    )
    resp = plan_seal_inventory(req)
    assert [item.seal_id for item in resp.excluded_inventory] == ["S9"]
    assert resp.results[0].status == "assigned"
    assert resp.results[0].seal_id == "S7"


def test_cross_chain_conflicting_seal_is_still_excluded():
    # 跨链冲突只在批量结果中标记；两个有效占用事件都排除该编号
    chains = [
        chain("c1", [ev(EventType.SEAL, 8, seal_id="X1", custodian="Alice")]),
        chain("c2", [ev(EventType.SEAL, 9, seal_id="X1", custodian="Bob")]),
    ]
    req = make_request(
        chains=chains,
        records=[record("Alice", dt(10), chain_id="c1")],
        seals=[seal("X1", 12, custodians=("Alice", "Bob"))],
    )
    resp = plan_seal_inventory(req)

    assert resp.batch_validation.valid is False
    assert [c.chain_id for c in resp.batch_validation.conflicts] == ["c1", "c2"]
    excluded = {item.seal_id: item for item in resp.excluded_inventory}
    assert set(excluded) == {"X1"}
    assert [(o.chain_id, o.first_chain_id, o.event_index) for o in excluded["X1"].occupied_by] == [
        ("c1", "c1", 0),
        ("c2", "c1", 0),
    ]
    assert resp.results[0].reason_code == "no_candidate"


def test_seal_in_violation_event_is_not_occupied():
    # 未封装直接 seal 之后又 seal：第二个 seal 违规，不构成占用
    chains = [
        chain("c1", [
            ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
            ev(EventType.SEAL, 9, seal_id="S2", custodian="Alice"),
        ]),
    ]
    req = make_request(
        chains=chains,
        records=[record("Alice", dt(10), chain_id="c1")],
        seals=[seal("S2", 12)],
    )
    resp = plan_seal_inventory(req)
    assert resp.batch_validation.valid is False
    assert resp.excluded_inventory == []
    assert resp.results[0].status == "assigned"
    assert resp.results[0].seal_id == "S2"


def test_excluded_inventory_sorted_by_seal_id():
    chains = [
        chain("c1", [
            ev(EventType.SEAL, 7, seal_id="S2", custodian="Alice"),
            ev(EventType.HANDOVER, 7, 30, custodian="Bob"),
        ]),
        chain("c2", [ev(EventType.SEAL, 8, seal_id="S1", custodian="Carol")]),
    ]
    req = make_request(
        chains=chains,
        records=[record("Alice", dt(10), chain_id="c1")],
        seals=[seal("S2", 12), seal("S1", 12)],
    )
    resp = plan_seal_inventory(req)
    # 排除项按 seal_id 字典序，不按占用发现顺序
    assert [item.seal_id for item in resp.excluded_inventory] == ["S1", "S2"]


# ---------------------------------------------------------------------------
# 全局最优：最大满足数 → 最小总余量 → 记录顺序 + seal_id 字典序（非逐条贪心）
# ---------------------------------------------------------------------------


def test_competition_one_loser_gets_winner_detail():
    req = make_request(
        chains=[chain("c1")],
        records=[record("Alice", dt(10)), record("Alice", dt(10))],
        seals=[seal("S1", 12)],
    )
    resp = plan_seal_inventory(req)
    results = by_index(resp)

    assert results[0].status == "assigned" and results[0].seal_id == "S1"
    loser = results[1]
    assert loser.status == "unassigned"
    assert loser.reason_code == "competition"
    assert loser.candidate_seal_ids == ["S1"]
    assert loser.competition[0].seal_id == "S1"
    assert loser.competition[0].assigned_record_index == 0
    assert resp.summary.competition == 1


def test_maximize_count_not_greedy_per_record():
    # r0（Bob）仅能取 S1，r1（Alice）可取 S1/S2。逐条贪心若在 r1 处先抢占 S1，
    # r0 便无候选，只满足 1 条；全局最优必须满足 2 条：r0=S1、r1=S2。
    req = SealInventoryPlanRequest(
        chains=[chain("c1")],
        records=[
            record("Bob", dt(10)),
            record("Alice", dt(10)),
        ],
        seals=[
            seal("S1", 11, custodians=("Alice", "Bob")),
            seal("S2", 12, custodians=("Alice",)),
        ],
    )
    resp = plan_seal_inventory(req)

    results = by_index(resp)
    assert results[0].status == "assigned" and results[0].seal_id == "S1"
    assert results[1].status == "assigned" and results[1].seal_id == "S2"
    assert resp.summary.assigned == 2


def test_minimize_total_slack_among_max_cardinality():
    # 两条记录、三枚封签，最大满足数恒为 2；枚举 2 条匹配的总余量：
    #   r0(09:00) 对 S1(13:00)=14400s、S2(11:00)=7200s；S3 不允许 r0
    #   r1(10:30) 对 S1=9000s、S2=1800s、S3(11:30)=3600s
    #   r0=S1,r1=S2=16200；r0=S1,r1=S3=18000；r0=S2,r1=S1=16200；
    #   r0=S2,r1=S3=10800  <- 总余量最小
    req = SealInventoryPlanRequest(
        chains=[chain("c1")],
        records=[record("Alice", dt(9)), record("Alice", dt(10, 30))],
        seals=[
            seal("S1", 13),
            seal("S2", 11),
            seal("S3", 11, expires_minute=30),
        ],
    )
    resp = plan_seal_inventory(req)
    results = by_index(resp)
    assert results[0].seal_id == "S2"
    assert results[1].seal_id == "S3"
    assert resp.summary.total_slack_seconds == 10800.0


def test_tie_break_record_order_before_seal_lex_order():
    # 仅 1 枚封签、两条同候选记录：记录 0 优先（输入顺序）
    req = make_request(
        chains=[chain("c1")],
        records=[
            record("Alice", dt(10)),
            record("Alice", dt(10)),
        ],
        seals=[seal("S9", 12)],
    )
    resp = plan_seal_inventory(req)
    results = by_index(resp)
    assert results[0].status == "assigned"
    assert results[1].reason_code == "competition"


def test_tie_break_seal_id_lex_when_count_and_slack_tie():
    # 两条记录只共享一个边集合不现实；构造：r0 有 S1/S2 两边且余量相同，
    # r1 无候选。最大化满足数=1，余量相同，字典序要求 r0 取 S1。
    req = make_request(
        chains=[chain("c1")],
        records=[
            record("Alice", dt(10)),
            record("Bob", dt(10)),
        ],
        seals=[
            seal("S2", 12, custodians=("Alice",)),
            seal("S1", 12, custodians=("Alice",)),
        ],
    )
    resp = plan_seal_inventory(req)
    results = by_index(resp)
    assert results[0].status == "assigned" and results[0].seal_id == "S1"
    assert results[1].reason_code == "no_candidate"


def test_no_per_record_greedy_lex_can_reduce_count():
    # 经典反例：若按输入顺序给每条记录选字典序最小的可行封签，
    # r0 会拿 S1，r1 仅剩 S2 不可用 → 只满足 1 条；
    # 全局最优满足 2 条：r0=S2, r1=S1。
    req = make_request(
        chains=[chain("c1")],
        records=[
            record("Alice", dt(10)),
            record("Bob", dt(10)),
        ],
        seals=[
            seal("S1", 12, custodians=("Alice", "Bob")),
            seal("S2", 12, custodians=("Alice",)),
        ],
    )
    resp = plan_seal_inventory(req)
    results = by_index(resp)
    assert results[0].status == "assigned" and results[0].seal_id == "S2"
    assert results[1].status == "assigned" and results[1].seal_id == "S1"
    assert resp.summary.assigned == 2


def test_results_keep_input_order_and_echo_fields():
    req = make_request(
        chains=[chain("c1"), chain("c2")],
        records=[
            record("Alice", dt(10), chain_id="c2"),
            record("Bob", dt(11), chain_id="c1"),
        ],
        seals=[
            seal("S1", 12, custodians=("Alice", "Bob")),
            seal("S2", 13, custodians=("Alice", "Bob")),
        ],
    )
    resp = plan_seal_inventory(req)
    assert [r.index for r in resp.results] == [0, 1]
    assert resp.results[0].chain_id == "c2" and resp.results[0].custodian == "Alice"
    assert resp.results[1].chain_id == "c1" and resp.results[1].use_at == dt(11)


# ---------------------------------------------------------------------------
# 422 语义错误
# ---------------------------------------------------------------------------


def test_duplicate_inventory_seal_id_rejected_by_model():
    with pytest.raises(ValidationError) as exc_info:
        make_request(
            chains=[chain("c1")],
            seals=[seal("S1", 12), seal("S1", 13)],
        )
    assert "库存封签编号不可重复" in str(exc_info.value)


def test_duplicate_chain_id_rejected_by_model():
    with pytest.raises(ValidationError):
        make_request(chains=[chain("dup"), chain("dup")])


def test_record_referencing_unknown_chain_raises_422():
    req = make_request(
        chains=[chain("c1")],
        records=[record(chain_id="ghost")],
        seals=[seal("S1", 12)],
    )
    with pytest.raises(SealInventoryPlanError, match="chain_id 不存在"):
        plan_seal_inventory(req)


def test_mixed_timezone_use_at_and_expires_at_raises_422():
    req = SealInventoryPlanRequest.model_validate({
        "chains": [{"chain_id": "c1", "events": []}],
        "records": [
            {"chain_id": "c1", "custodian": "Alice", "use_at": f"{DAY}T10:00:00Z"},
        ],
        "seals": [
            {"seal_id": "S1", "expires_at": f"{DAY}T12:00:00",
             "allowed_custodians": ["Alice"]},
        ],
    })
    # 模型层允许各自合法的时间；混用在业务层报 422
    with pytest.raises(SealInventoryPlanError, match="时区信息不一致"):
        plan_seal_inventory(req)


def test_mixed_timezone_with_chain_events_raises_422():
    req = SealInventoryPlanRequest.model_validate({
        "chains": [
            {"chain_id": "c1", "events": [
                {"type": "seal", "timestamp": f"{DAY}T08:00:00",
                 "seal_id": "X", "custodian": "Alice"},
            ]},
        ],
        "records": [
            {"chain_id": "c1", "custodian": "Alice", "use_at": f"{DAY}T10:00:00Z"},
        ],
        "seals": [
            {"seal_id": "S1", "expires_at": f"{DAY}T12:00:00Z",
             "allowed_custodians": ["Alice"]},
        ],
    })
    with pytest.raises(SealInventoryPlanError, match="时区信息不一致"):
        plan_seal_inventory(req)


def test_empty_allowed_custodians_rejected_by_model():
    payload = {
        "chains": [],
        "records": [],
        "seals": [
            {"seal_id": "S1", "expires_at": f"{DAY}T12:00:00Z",
             "allowed_custodians": []},
        ],
    }
    with pytest.raises(ValidationError):
        SealInventoryPlanRequest.model_validate(payload)


# ---------------------------------------------------------------------------
# API 集成
# ---------------------------------------------------------------------------


def test_api_full_flow():
    payload = {
        "chains": [
            {"chain_id": "c1", "events": [
                {"type": "seal", "timestamp": f"{DAY}T08:00:00Z",
                 "seal_id": "USED", "custodian": "Alice"},
            ]},
        ],
        "records": [
            {"chain_id": "c1", "custodian": "Alice", "use_at": f"{DAY}T10:00:00Z"},
            {"chain_id": "c1", "custodian": "Alice", "use_at": f"{DAY}T10:00:00Z"},
        ],
        "seals": [
            {"seal_id": "USED", "expires_at": f"{DAY}T18:00:00Z",
             "allowed_custodians": ["Alice"]},
            {"seal_id": "S2", "expires_at": f"{DAY}T12:00:00Z",
             "allowed_custodians": ["Alice"]},
            {"seal_id": "S1", "expires_at": f"{DAY}T13:00:00Z",
             "allowed_custodians": ["Alice"]},
        ],
    }
    r = client.post("/api/v1/seal-chain/seal-inventory-plan", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert [item["seal_id"] for item in data["excluded_inventory"]] == ["USED"]
    assert data["results"][0]["status"] == "assigned"
    assert data["results"][0]["seal_id"] == "S1"
    assert data["results"][1]["seal_id"] == "S2"
    assert data["summary"]["assigned"] == 2
    assert data["summary"]["total_slack_seconds"] == 18000.0
    # 批量校验结果原样保留
    assert data["batch_validation"]["chains"][0]["chain_id"] == "c1"
    assert data["batch_validation"]["valid"] is True


def test_api_duplicate_inventory_seal_returns_422():
    payload = {
        "chains": [],
        "records": [],
        "seals": [
            {"seal_id": "S1", "expires_at": f"{DAY}T11:00:00Z",
             "allowed_custodians": ["Alice"]},
            {"seal_id": "S1", "expires_at": f"{DAY}T12:00:00Z",
             "allowed_custodians": ["Alice"]},
        ],
    }
    assert client.post(
        "/api/v1/seal-chain/seal-inventory-plan", json=payload
    ).status_code == 422


def test_api_mixed_timezone_returns_422():
    payload = {
        "chains": [],
        "records": [
            {"chain_id": "c1", "custodian": "Alice", "use_at": f"{DAY}T10:00:00Z"},
        ],
        "seals": [
            {"seal_id": "S1", "expires_at": f"{DAY}T12:00:00",
             "allowed_custodians": ["Alice"]},
        ],
    }
    assert client.post(
        "/api/v1/seal-chain/seal-inventory-plan", json=payload
    ).status_code == 422


def test_api_unknown_chain_reference_returns_422():
    payload = {
        "chains": [],
        "records": [
            {"chain_id": "nope", "custodian": "Alice", "use_at": f"{DAY}T10:00:00Z"},
        ],
        "seals": [],
    }
    assert client.post(
        "/api/v1/seal-chain/seal-inventory-plan", json=payload
    ).status_code == 422
