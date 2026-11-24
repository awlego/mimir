# Test job: counts to 120 slowly, checkpointing every 10 steps; resumes where it stopped. No GPU.
exec python infra/gaming-pc/dummy_job.py runs/dummy 120
