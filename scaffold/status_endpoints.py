"""
status_endpoints.py
===================

FastAPI router with all status/monitoring endpoints.

Import this into web_api_monetized.py:

    from status_endpoints import make_status_router
    status_router = make_status_router(scaffold, collective)
    app.include_router(status_router)

Provides:
    GET /status       — system overview
    GET /workspace    — DMN global workspace
    GET /beliefs      — belief system state  
    GET /insights     — spontaneous insights
    GET /personality  — personality traits
    GET /progress     — improvement over time
    GET /training     — small model status
    GET /autonomous   — self-questioning activity log
"""

from fastapi import APIRouter
from typing import Optional


def make_status_router(scaffold: dict, collective=None) -> APIRouter:
    router = APIRouter()

    def sc(key, default=None):
        return scaffold.get(key, default) if scaffold else default

    @router.get("/status")
    async def status():
        if not scaffold:
            return {"ready": False}
        base = {"ready": True}
        try:
            dmn = sc("dmn")
            if dmn:
                base["dmn"] = {
                    "phi":    round(dmn.mean_phi(), 4),
                    "trend":  dmn.phi_trend(),
                    "cycles": dmn._cycle_count,
                    "emotion": dmn.workspace.dominant_emotion,
                    "engagement": round(dmn.workspace.engagement_level, 2),
                }
        except Exception: pass
        try:
            if collective:
                base["collective"] = collective.stats()
        except Exception: pass
        try:
            fs = sc("from_scratch")
            if fs:
                base["training"] = {
                    "stage":    fs._stage,
                    "examples": len(fs._examples),
                    "backend":  fs._backend,
                }
        except Exception: pass
        try:
            from progress_tracker import get_tracker
            tracker = get_tracker()
            if tracker:
                base["progress"] = tracker.latest_snapshot()
        except Exception: pass
        try:
            from autonomous_loop import get_loop
            loop = get_loop()
            if loop:
                base["autonomous"] = loop.status()
        except Exception: pass
        return base

    @router.get("/workspace")
    async def workspace():
        dmn = sc("dmn")
        if not dmn:
            return {}
        try:
            return dmn.workspace.to_dict()
        except Exception:
            return {}

    @router.get("/beliefs")
    async def beliefs():
        plastic = sc("plastic")
        if not plastic:
            return {"questioning": []}
        try:
            q = plastic.beliefs.questioning_beliefs()
            all_b = plastic.beliefs.all_beliefs()
            return {
                "questioning": [
                    {"statement": b.statement[:80],
                     "confidence": round(b.confidence, 3),
                     "times_failed": b.times_failed}
                    for b in q[:5]
                ],
                "strongest": [
                    {"statement": b.statement[:80], "confidence": round(b.confidence, 3)}
                    for b in sorted(all_b, key=lambda x: x.confidence, reverse=True)[:3]
                ],
                "total": len(all_b),
            }
        except Exception:
            return {"questioning": []}

    @router.get("/insights")
    async def insights():
        dmn = sc("dmn")
        if not dmn:
            return {"insights": []}
        try:
            today = dmn.spontaneous_insights_today()
            log = [
                e.get("cross_domain", "")
                for e in (dmn._activity_log or [])[-20:]
                if e.get("cross_domain")
            ]
            return {"insights": today[-10:], "activity_log": log[-5:]}
        except Exception:
            return {"insights": []}

    @router.get("/personality")
    async def personality():
        persona = sc("persona")
        if not persona:
            return {"traits": []}
        try:
            return {
                "traits": [
                    {"name": n, "strength": round(t["strength"], 3),
                     "description": t.get("description", "")}
                    for n, t in sorted(
                        persona.traits.items(),
                        key=lambda x: -x[1]["strength"]
                    )
                ],
                "observations": persona._n_observations,
            }
        except Exception:
            return {"traits": []}

    @router.get("/progress")
    async def progress():
        try:
            from progress_tracker import get_tracker
            tracker = get_tracker()
            if tracker:
                return tracker.full_history()
        except Exception:
            pass
        return {"snapshots": [], "message": "progress_tracker not initialised"}

    @router.get("/training")
    async def training():
        fs = sc("from_scratch")
        if not fs:
            return {"available": False}
        try:
            import os, json
            result = {
                "available": True,
                "stage": fs._stage,
                "examples": len(fs._examples),
                "backend": fs._backend,
                "model_dir": fs.output_dir,
                "model_exists": os.path.exists(
                    os.path.join(fs.output_dir, "model")
                ),
            }
            comp_path = os.path.join(fs.output_dir, "competency.json")
            if os.path.exists(comp_path):
                comp = json.load(open(comp_path))
                records = comp.get("records", [])
                if records:
                    last = records[-1]
                    result["last_test"] = {
                        "stage":       last.get("stage"),
                        "verify_rate": round(last.get("verify_rate", 0), 3),
                        "parse_rate":  round(last.get("parse_rate", 0), 3),
                        "examples":    last.get("n_examples_trained", 0),
                    }
                    result["history"] = [
                        {"verify_rate": round(r.get("verify_rate",0),3),
                         "examples":    r.get("n_examples_trained",0)}
                        for r in records[-20:]
                    ]
            return result
        except Exception as e:
            return {"available": True, "error": str(e)}

    @router.get("/autonomous")
    async def autonomous():
        try:
            from autonomous_loop import get_loop
            loop = get_loop()
            if loop:
                return loop.full_status()
        except Exception:
            pass
        return {"running": False, "message": "autonomous loop not started"}

    return router

    @router.get("/meta")
    async def meta_status():
        try:
            from meta_optimizer import get_meta
            meta = get_meta()
            if meta:
                return meta.status()
        except Exception:
            pass
        return {"running": False, "message": "meta optimizer not started"}

