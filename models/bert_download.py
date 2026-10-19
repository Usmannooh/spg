
import os


from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="emilyalsentzer/Bio_ClinicalBERT",
    local_dir=r"E:\usman\GOU1\spg\Bio_ClinicalBERT",
    resume_download=True,
    force_download=False
)

