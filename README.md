# OSOW Damage Calculator — Streamlit Web App

## Folder Structure Required

```
your_folder/
├── app.py                  ← this file
├── requirements.txt
├── tables.py               ← your existing tables file
└── C33/
    ├── weights-mn/
    │   ├── ANN_model_SABU_Climate33Base1Slab1Shoulder1.h5
    │   ├── ANN_model_TABU_Climate33Base1Slab1Shoulder1.h5
    │   ├── ANN_model_TATD_Climate33Base1Slab1Shoulder1.h5
    │   └── ... (all combinations)
    └── scalers-mn/
        ├── Min_Max_scaler_X_Climate33Base1Slab1Shoulder1.pkl
        ├── Min_Max_scaler_FD_Climate33Base1Slab1Shoulder1.pkl
        └── ... (all combinations)
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Deploy to Streamlit Cloud (Free)

1. Push this folder to a GitHub repository
2. Go to https://share.streamlit.io
3. Connect your GitHub account
4. Select your repo and app.py
5. Click Deploy
6. Share the link with anyone

## Deploy to Render.com (Free)

1. Push to GitHub
2. Go to https://render.com
3. New → Web Service → connect your repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
6. Deploy

## Notes

- The **Model files base path** in the sidebar must point to the folder
  containing your C33/ directory.
- On Streamlit Cloud, all model files must be in the repository.
- If model files are too large for GitHub (>100MB each), use Git LFS.
