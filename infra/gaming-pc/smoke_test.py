"""Gaming-pc smoke test: CUDA sees an sm_120 GPU, and Gemma-3-4B-IT loads in bf16 and generates.

Run from the repo root:  env3.12/bin/python infra/gaming-pc/smoke_test.py
"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "google/gemma-3-4b-it"

assert torch.cuda.is_available(), "CUDA not available: see SETUP.md step 4 troubleshooting"
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "|", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0), "| arch list", torch.cuda.get_arch_list())
assert "sm_120" in torch.cuda.get_arch_list(), "this torch build has no sm_120 kernels: reinstall (SETUP.md step 4)"
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", (x @ x).float().abs().mean().item() > 0)

t = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
print(f"loaded {type(model).__name__} in {time.time() - t:.0f}s")

messages = [{"role": "user", "content": "In one sentence, what is a model organism in biology?"}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
t = time.time()
out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
new = out[0, inputs["input_ids"].shape[1]:]
print(f"generated {len(new)} tokens in {time.time() - t:.1f}s:\n  {tok.decode(new, skip_special_tokens=True).strip()}")
print(f"VRAM: {torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated, "
      f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB peak, "
      f"of {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
print("SMOKE TEST PASSED")
