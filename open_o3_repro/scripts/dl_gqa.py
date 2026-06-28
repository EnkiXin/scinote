from huggingface_hub import snapshot_download
snapshot_download(repo_id="lmms-lab/GQA", repo_type="dataset",
    local_dir="/home/yz0392@unt.ad.unt.edu/xin_ai/open_o3/data/source_datasets/GQA", max_workers=8)
print("GQA DONE")
