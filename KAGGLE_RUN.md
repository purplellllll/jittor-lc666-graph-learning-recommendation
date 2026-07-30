# Kaggle run notes

Use a Kaggle Notebook with GPU and Internet enabled.

1. Upload this folder, or at least these files, into `/kaggle/working`:
   - `gcn.py`
   - `kaggle_install_and_run.sh`

   `gcn.py` will download the official starter package automatically if
   `data/cora.pkl` is not already present.

2. Run:

```bash
cd /kaggle/working
bash kaggle_install_and_run.sh
```

3. Submit `/kaggle/working/result.zip`.

Expected archive layout:

```text
result.zip
  gcn.py
  result.json
```

If Kaggle Internet is disabled, enable it in notebook settings before installing
Jittor and JittorGeometric.

For API automation from this machine, put your Kaggle token at
`%USERPROFILE%\.kaggle\kaggle.json`, then run:

```powershell
.\push_kaggle.ps1
kaggle kernels status <your-username>/jittor-warmup1-cora-gcn
kaggle kernels output <your-username>/jittor-warmup1-cora-gcn -p kaggle_output --force
```
