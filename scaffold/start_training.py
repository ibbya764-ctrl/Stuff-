"""
start_training.py — Check and start 500MB model training
Run: python start_training.py
"""
import os,sys,json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
BASE="./scaffold_data"; DATA_DIR=BASE+"/scratch_data"; MODEL_DIR=BASE+"/small_model"
DATA_FILE=DATA_DIR+"/all_examples.jsonl"

def count_examples():
    if not os.path.exists(DATA_FILE): return 0
    try:
        with open(DATA_FILE) as f: return sum(1 for l in f if l.strip())
    except: return 0

def check_pytorch():
    try:
        import torch
        if torch.cuda.is_available(): return "cuda","NVIDIA GPU: "+torch.cuda.get_device_name(0)
        mps=getattr(torch.backends,"mps",None)
        if mps and torch.backends.mps.is_available(): return "mps","Apple Silicon GPU"
        return "cpu",f"CPU ({os.cpu_count()} cores)"
    except ImportError: return None,"PyTorch not installed"

def print_status():
    print("\n"+"="*55+"\n  Scaffold — Training Status\n"+"="*55+"\n")
    n=count_examples()
    print(f"Training examples collected: {n}")
    if n<20: print(f"  Need {20-n} more verified questions before first training cycle")
    elif n<50: print(f"  Training will start — more examples = better results")
    else: print(f"  Good amount — training will be meaningful")
    device,desc=check_pytorch()
    print(f"\nPyTorch: {desc}")
    if device is None:
        print("\n  Install for CPU training on your Lenovo:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("  pip install transformers")
    elif device=="cpu":
        print("  Training on CPU: ~5-15 sec/step, ~30-90 min/epoch. Good for overnight.")
    model_exists=os.path.exists(MODEL_DIR+"/model")
    print(f"\nSmall model: {'YES — saved at '+MODEL_DIR+'/model' if model_exists else 'Not yet trained'}")
    comp=MODEL_DIR+"/competency.json"
    if os.path.exists(comp):
        try:
            recs=json.load(open(comp)).get("records",[])
            if recs:
                last=recs[-1]
                print(f"\nLast test: stage={last.get('stage')} verify={last.get('verify_rate',0):.0%} parse={last.get('parse_rate',0):.0%}")
        except: pass
    print()

def run_training():
    n=count_examples()
    if n<5: print(f"\nOnly {n} examples. Run mac_run.py and ask questions first."); return
    device,_=check_pytorch()
    if device is None: print("\nInstall PyTorch first (see above)"); return
    try:
        import transformers
    except ImportError: print("\npip install transformers"); return
    print(f"\nStarting training on {device.upper()} with {n} examples...\n")
    try:
        from from_scratch_trainer import FromScratchTrainer
        t=FromScratchTrainer(base_model_name="Qwen/Qwen2-0.5B",output_dir=MODEL_DIR,data_dir=DATA_DIR,verbose=True)
        print(f"Backend: {t._backend}")
        if len(t._examples)<5: t._load_examples()
        if len(t._examples)<5: print("Not enough examples."); return
        t._run_training_cycle()
        print(f"\nDone. Model saved to {MODEL_DIR}/model")
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you have ~8GB RAM free and transformers installed.")

if __name__=="__main__":
    print_status()
    n=count_examples()
    if n>=5:
        if input("Start a training cycle now? (y/n): ").strip().lower()=='y': run_training()
    else:
        print("Run mac_run.py and ask questions to collect training examples.")
        print("Training triggers automatically after 20 verified answers.")
