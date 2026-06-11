"""
web_api.py — open access version
No login required. IP-based rate limiting only.
Set SCAFFOLD_SECRET for admin endpoints.
"""
import os,sys,json,time,asyncio
from typing import Optional

try:
    from fastapi import FastAPI,WebSocket,WebSocketDisconnect,HTTPException,Depends,Header,Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
    import uvicorn
    HAS_FASTAPI=True
except ImportError:
    HAS_FASTAPI=False

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))

SCAFFOLD_SECRET=os.environ.get("SCAFFOLD_SECRET","change-this-for-admin")
RATE_HOUR=int(os.environ.get("RATE_LIMIT_HOUR","30"))
RATE_DAY=int(os.environ.get("RATE_LIMIT_DAY","200"))
MAX_CONCURRENT=int(os.environ.get("MAX_CONCURRENT","2"))

def llm_chat(system,user):
    import requests
    try:
        r=requests.post("http://localhost:11434/api/chat",json={
            "model":"qwen2.5:7b",
            "messages":[{"role":"system","content":system},{"role":"user","content":user}],
            "stream":False,"options":{"temperature":0.7,"num_predict":2000}},timeout=120)
        if r.status_code==200: return r.json()["message"]["content"]
    except Exception: pass
    key=os.environ.get("ANTHROPIC_API_KEY","")
    if key:
        r=requests.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key":key,"anthropic-version":"2023-06-01","content-type":"application/json"},
            json={"model":"claude-sonnet-4-20250514","max_tokens":2000,"system":system,
                  "messages":[{"role":"user","content":user}]},timeout=60)
        if r.status_code==200: return r.json()["content"][0]["text"]
    raise RuntimeError("No LLM available")

def load_scaffold():
    try:
        from enhanced_pipeline import EnhancedConfig
        from thought_stream import ThoughtAwareBrainPipeline
        from brain_modules import BrainModulesBundle
        from continual_trainer import ContinualScaffoldTrainer
        from from_scratch_trainer import FromScratchTrainer
        from psychology_extended import FullPsychologicalBundle
        from belief_system import PlasticPsychologicalCore
        from psych_modules import EpistemicHumilityCalibrator,PersonalityConsistency,PsychologicalIntegration
        from default_mode_network import DefaultModeNetwork
        from collective_learning import CollectiveLearningHub
        from knowledge_access import KnowledgeOrchestrator
        from compute_router import ComputeRouter

        BASE,SHARED="./scaffold_data","./shared_data"
        brain=BrainModulesBundle(base_dir=BASE)
        psych=FullPsychologicalBundle(base_dir=BASE)
        plastic=PlasticPsychologicalCore(base_dir=BASE)
        calib=EpistemicHumilityCalibrator(os.path.join(BASE,"calib.json"))
        persona=PersonalityConsistency(os.path.join(BASE,"personality.json"))
        integr=PsychologicalIntegration(os.path.join(BASE,"integration.json"))
        know=KnowledgeOrchestrator(base_dir=BASE)
        router=ComputeRouter(os.path.join(BASE,"compute_router.json"),verbose=False)
        pipeline=ThoughtAwareBrainPipeline(llm_chat_fn=llm_chat,
            config=EnhancedConfig(base_dir=BASE,domain_name="general",verbose=False),
            enable_stream=False,enable_self_review=True,enable_verify_loop=True,verbose=False)
        episodic=trainer=fs=None
        try:
            episodic=pipeline._base._enhanced._episodic
            trainer=ContinualScaffoldTrainer(episodic_store=episodic,base_dir=BASE,
                model_name="Qwen/Qwen2.5-7B-Instruct",trigger_every=5,verbose=False)
            fs=FromScratchTrainer(base_model_name="Qwen/Qwen2-0.5B",
                output_dir=os.path.join(BASE,"small_model"),
                data_dir=os.path.join(BASE,"scratch_data"),verbose=False)
        except Exception: pass
        mods={"brain":brain,"psych":psych,"plastic":plastic,"episodic":episodic}
        dmn=DefaultModeNetwork(mods,llm_chat_fn=llm_chat,cycle_seconds=45,base_dir=BASE,verbose=False)
        dmn.start()
        collective=CollectiveLearningHub(base_dir=SHARED,verbose=False)
        print("[scaffold] Ready.")
        return {"llm_chat":llm_chat,"pipeline":pipeline,"brain":brain,"psych":psych,
                "plastic":plastic,"calib":calib,"persona":persona,"integr":integr,
                "knowledge":know,"router":router,"dmn":dmn,"trainer":trainer,
                "from_scratch":fs,"episodic":episodic,"collective":collective}
    except Exception as e:
        print(f"[scaffold] Load error: {e}"); return None

def process_question(question,domain,ip,sc,collective):
    if not sc: return {"response":"The system is still loading. Try again in a moment.","verified":False}
    sc["dmn"].pause_for_task()
    try:
        ctx="\n\n".join(c for c in [
            sc["brain"].pre_reasoning_context(question,domain),
            sc["psych"].pre_reasoning_context(question,domain),
            sc["plastic"].pre_reasoning_context(question,domain),
            sc["knowledge"].pre_reasoning_context(question,domain),
            sc["dmn"].get_context(),
            collective.format_shared_context(question,domain),
        ] if c.strip())
        auto=sc["brain"].try_automatic(question,domain)
        if auto:
            _,ar=auto; sc["brain"].post_reasoning_update(ar,question,domain)
            return {"response":ar.result,"verified":True,"confidence":ar.confidence,"automatic":True,"phi":sc["dmn"].workspace.phi_estimate}
        result=sc["pipeline"].ask(f"{question}\n\n{ctx}" if ctx else question,domain=domain)
        r=result.get("reasoning"); resp=result.get("response","")
        if r:
            sc["brain"].post_reasoning_update(r,question,domain,success=r.verified)
            fb=sc["psych"].evaluate(r,resp,question)
            sc["plastic"].process_run(r,resp,question)
            sc["calib"].record_claim(r.confidence,r.verified,"derived",domain)
            sc["persona"].update_from_response(resp)
            integ,_=sc["integr"].compute_integration({"values":fb.value_score,"aesthetic":fb.aesthetic_score,"social":fb.social_score,"engagement":fb.engagement})
            sc["dmn"].integrate_task_result(result,question)
            if r.verified:
                collective.contribute(r,question,ip,{"messages":[
                    {"role":"user","content":question},{"role":"assistant","content":resp}]})
            if sc.get("trainer") and sc.get("episodic"):
                try:
                    rec=sc["episodic"].query_recent(n=1)
                    if rec: sc["trainer"].on_run_complete(result,rec[0])
                except Exception: pass
            if sc.get("from_scratch") and r.verified: sc["from_scratch"].add_verified_run(r,question)
            collective.update_session_activity(ip,r.verified)
            return {"response":resp,"verified":r.verified,"confidence":r.confidence,
                    "quality":fb.internal_quality,"phi":sc["dmn"].workspace.phi_estimate,
                    "emotion":fb.emotional_state}
        return {"response":resp,"verified":False}
    finally: sc["dmn"].resume()

if HAS_FASTAPI:
    app=FastAPI(title="Scaffold AI")
    app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
    scaffold=collective_hub=None
    ip_rate:dict={}
    processing=0

    def check_rate(ip):
        now=time.time()
        h=ip_rate.setdefault(ip,[])
        ip_rate[ip]=[t for t in h if t>now-3600]
        if len(ip_rate[ip])>=RATE_HOUR: return False,"Rate limit: max "+str(RATE_HOUR)+" questions/hour"
        if processing>=MAX_CONCURRENT: return False,"Server busy. Please wait a moment."
        return True,""

    def record_ip(ip):
        ip_rate.setdefault(ip,[]).append(time.time())
        global processing; processing=min(MAX_CONCURRENT,processing+1)

    def release_ip():
        global processing; processing=max(0,processing-1)

    def check_admin(authorization:str=Header(None)):
        if not authorization or authorization.replace("Bearer ","")!=SCAFFOLD_SECRET:
            raise HTTPException(403,"Admin key required")
        return True

    @app.on_event("startup")
    async def startup():
        global scaffold,collective_hub
        loop=asyncio.get_event_loop()
        scaffold=await loop.run_in_executor(None,load_scaffold)
        collective_hub=scaffold["collective"] if scaffold else None

    class AskReq(BaseModel): question:str; domain:str="general"

    @app.post("/ask")
    async def ask(req:AskReq,request:Request):
        ip=request.client.host
        ok,reason=check_rate(ip)
        if not ok: raise HTTPException(429,reason)
        record_ip(ip)
        try:
            loop=asyncio.get_event_loop()
            result=await loop.run_in_executor(None,process_question,
                req.question,req.domain,ip,scaffold,collective_hub)
            return result
        finally: release_ip()

    @app.get("/status")
    async def status():
        base={"ready":scaffold is not None,"processing":processing,"max_concurrent":MAX_CONCURRENT}
        if scaffold:
            base["dmn"]={"phi":round(scaffold["dmn"].mean_phi(),4),"trend":scaffold["dmn"].phi_trend(),"cycles":scaffold["dmn"]._cycle_count}
            base["collective"]=scaffold["collective"].stats()
            fs=scaffold.get("from_scratch")
            if fs: base["training"]={"stage":fs._stage,"examples":len(fs._examples),"backend":fs._backend}
        return base

    @app.get("/workspace")
    async def workspace():
        if not scaffold: return {}
        return scaffold["dmn"].workspace.to_dict()

    @app.get("/beliefs")
    async def beliefs():
        if not scaffold: return {}
        q=[{"statement":b.statement[:80],"confidence":b.confidence,"times_failed":b.times_failed}
           for b in scaffold["plastic"].beliefs.questioning_beliefs()]
        return {"questioning":q}

    @app.get("/personality")
    async def personality():
        if not scaffold: return {}
        return {"traits":[{"name":n,"strength":round(t["strength"],3),"description":t["description"]}
                           for n,t in sorted(scaffold["persona"].traits.items(),key=lambda x:-x[1]["strength"])]}

    @app.get("/insights")
    async def insights():
        if not scaffold: return {}
        return {"insights":scaffold["dmn"].spontaneous_insights_today()[-10:]}

    @app.get("/admin/status",dependencies=[Depends(check_admin)])
    async def admin_status():
        s={"ip_counts":{ip:len(h) for ip,h in ip_rate.items()}}
        if scaffold:
            s["knowledge"]=scaffold["knowledge"].status()
            s["plastic"]=scaffold["plastic"].status()
            fs=scaffold.get("from_scratch")
            if fs: s["training_detail"]=fs.status()
        return s

    @app.websocket("/ws")
    async def ws(websocket:WebSocket):
        await websocket.accept()
        ip=websocket.client.host
        try:
            while True:
                data=await websocket.receive_json()
                q=data.get("question",""); d=data.get("domain","general")
                if not q: continue
                ok,reason=check_rate(ip)
                if not ok:
                    await websocket.send_json({"type":"error","content":reason}); continue
                record_ip(ip)
                await websocket.send_json({"type":"thinking","content":"Processing your question..."})
                try:
                    loop=asyncio.get_event_loop()
                    result=await loop.run_in_executor(None,process_question,q,d,ip,scaffold,collective_hub)
                    await websocket.send_json({"type":"response","content":result.get("response",""),
                        "verified":result.get("verified",False),"confidence":result.get("confidence","?"),
                        "phi":result.get("phi",0),"emotion":result.get("emotion","")})
                finally:
                    release_ip()
                await websocket.send_json({"type":"done"})
        except WebSocketDisconnect: pass
        except Exception as e:
            try: await websocket.send_json({"type":"error","content":str(e)})
            except Exception: pass

    static=os.path.join(os.path.dirname(__file__),"web_static")
    if os.path.exists(static): app.mount("/",StaticFiles(directory=static,html=True),name="static")

if __name__=="__main__":
    if not HAS_FASTAPI: print("pip install fastapi uvicorn")
    else: uvicorn.run("web_api:app",host="0.0.0.0",port=8000,reload=False)


# ── Integrated state wiring (appended) ───────────────────
# Import at top of process_question to use full integrated state
def process_question_integrated(question, domain, ip, sc, collective):
    """
    Full integrated state version of process_question.
    Replace the call in /ask and /ws with this function
    once integrated_state.py and the updated language_module.py are in place.
    """
    if not sc: return {"response":"The system is still loading.","verified":False}

    try:
        from integrated_state import StateAssembler
        from language_module import LanguageModule
        assembler = StateAssembler(
            brain=sc.get("brain"), psych=sc.get("psych"),
            plastic=sc.get("plastic"), dmn=sc.get("dmn"),
            persona=sc.get("persona"), knowledge=sc.get("knowledge"),
        )
        lang = LanguageModule(sc["llm_chat"])
    except ImportError:
        return process_question(question, domain, ip, sc, collective)

    sc["dmn"].pause_for_task()
    try:
        # Assemble full state before reasoning
        state = assembler.assemble(question, domain)

        # Add collective context
        session = collective.get_session(ip)
        cctx = collective.format_shared_context(question, domain)

        # Build reasoning context from state
        ctx_parts = [
            sc["brain"].pre_reasoning_context(question, domain),
            sc["psych"].pre_reasoning_context(question, domain),
            sc["plastic"].pre_reasoning_context(question, domain),
            sc["knowledge"].pre_reasoning_context(question, domain),
            sc["dmn"].get_context(),
            cctx,
        ]
        ctx = "\n\n".join(c for c in ctx_parts if c.strip())

        # Procedural shortcut
        auto = sc["brain"].try_automatic(question, domain)
        if auto:
            _, ar = auto
            sc["brain"].post_reasoning_update(ar, question, domain)
            assembler.update_with_result(state, ar)
            output = lang.speak_from_state(state)
            return {"response": output.text, "verified": True,
                    "confidence": ar.confidence, "automatic": True,
                    "phi": sc["dmn"].workspace.phi_estimate,
                    "reasoning_trace": output.reasoning_trace}

        # Full pipeline
        result = sc["pipeline"].ask(f"{question}\n\n{ctx}" if ctx else question, domain=domain)
        r = result.get("reasoning")
        if r:
            # Update state with result
            assembler.update_with_result(state, r)

            # Generate natural output from full integrated state
            output = lang.speak_from_state(state)

            # Update all modules
            sc["brain"].post_reasoning_update(r, question, domain, success=r.verified)
            fb = sc["psych"].evaluate(r, output.text, question)
            sc["plastic"].process_run(r, output.text, question)
            sc["calib"].record_claim(r.confidence, r.verified, "derived", domain)
            sc["persona"].update_from_response(output.text)
            integ, _ = sc["integr"].compute_integration({
                "values": fb.value_score, "aesthetic": fb.aesthetic_score,
                "social": fb.social_score, "engagement": fb.engagement})
            sc["dmn"].integrate_task_result(result, question)

            if r.verified:
                collective.contribute(r, question, ip, {
                    "messages": [{"role":"user","content":question},
                                 {"role":"assistant","content":output.text}]})

            if sc.get("trainer") and sc.get("episodic"):
                try:
                    rec = sc["episodic"].query_recent(n=1)
                    if rec: sc["trainer"].on_run_complete(result, rec[0])
                except Exception: pass

            if sc.get("from_scratch") and r.verified:
                sc["from_scratch"].add_verified_run(r, question)

            collective.update_session_activity(ip, r.verified)

            return {"response": output.text, "verified": r.verified,
                    "confidence": r.confidence, "quality": fb.internal_quality,
                    "phi": sc["dmn"].workspace.phi_estimate,
                    "reasoning_trace": output.reasoning_trace,
                    "emotion": state.dominant_emotion}

        return {"response": result.get("response",""), "verified": False}
    finally:
        sc["dmn"].resume()
