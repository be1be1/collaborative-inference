from ray.job_submission import JobSubmissionClient
client = JobSubmissionClient("http://localhost:8265")
kick_off_benchmark = (
    # Clone ray. If ray is already present, don't clone again.
    #"git clone https://github.com/ray-project/ray || true;"
    # Run the benchmark.
    "python GoogLeNet.py"
    #" --size 100G --disable-check"
)
submission_id = client.submit_job(
    entrypoint=kick_off_benchmark,
    runtime_env={
        "working_dir": "/data/ray-cluster/procedure"
    }
)

print(submission_id)
