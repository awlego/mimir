"""A GPU-free job that follows the jobs/README.md contract: resume, checkpoint on SIGTERM, DONE marker."""
import json, os, signal, sys, time

out, total = sys.argv[1], int(sys.argv[2])
os.makedirs(out, exist_ok=True)
state_file = os.path.join(out, "state.json")
step = json.load(open(state_file))["step"] if os.path.exists(state_file) else 0
print(f"starting at step {step}/{total}", flush=True)

stop = False
def on_term(*_):
    global stop
    stop = True
signal.signal(signal.SIGTERM, on_term)

def checkpoint():
    with open(state_file + ".tmp", "w") as f:
        json.dump({"step": step}, f)
    os.replace(state_file + ".tmp", state_file)  # atomic

while step < total and not stop:
    time.sleep(1)
    step += 1
    if step % 10 == 0:
        checkpoint()
        print(f"step {step}/{total}, checkpointed", flush=True)
checkpoint()
if stop:
    print(f"SIGTERM: checkpointed at step {step}, exiting", flush=True)
    sys.exit(0)
json.dump({"steps": total}, open(os.path.join(out, "result.json"), "w"))
open(os.path.join(out, "DONE"), "w").close()
print("done", flush=True)
