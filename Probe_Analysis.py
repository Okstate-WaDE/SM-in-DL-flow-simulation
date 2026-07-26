
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import xarray as xr

from neuralhydrology.evaluation import get_tester
from neuralhydrology.utils.config import Config

# USER SETTINGS
run_dir = "/Users/gautam/Research/codesforresearch/percentsaturation/after first review/Voronoi/runs/no_soil_2502_110229"
basin = "Washita"

washita_nc_path = "/Users/gautam/Research/codesforresearch/percentsaturation/after first review/Voronoi/time_series/Washita.nc"
sm_var = "TR25"

batch_size = 256

#LOAD SOIL MOISTURE SERIES
ds_sm = xr.open_dataset(washita_nc_path)
sm_series = ds_sm[sm_var].to_series()
sm_series.index = pd.to_datetime(sm_series.index)
sm_series = sm_series.sort_index()

print("SM series range:", sm_series.index.min(), "to", sm_series.index.max())
print("SM series non-NaN:", sm_series.notna().sum())
print("SM dims:", ds_sm[sm_var].dims)

#COLLATE (handles datetime64)

def nh_collate(batch):
    out = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        v0 = vals[0]

        if torch.is_tensor(v0):
            out[k] = torch.stack(vals, dim=0)

        elif isinstance(v0, np.ndarray):
            if np.issubdtype(v0.dtype, np.datetime64):
                out[k] = vals  # keep list of arrays
            else:
                out[k] = torch.as_tensor(np.stack(vals, axis=0))

        elif isinstance(v0, np.datetime64):
            out[k] = vals

        else:
            out[k] = vals

    return out

#ROBUST FORWARD CALL
def run_forward(model, batch, device):
    x_d = batch["x_d"].to(device)

    # Option 1: model expects a batch dict
    try:
        return model({"x_d": x_d})
    except Exception:
        pass

# EXTRACT INPUT + CELL + END DATES
def extract_input_cell_and_end_dates(tester, basin, batch_size=256):
    model = tester.model
    model.eval()
    device = next(model.parameters()).device

    ds = tester._get_dataset(basin)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=nh_collate)

    # Find an LSTM-like module to hook
    lstm_module = None
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.LSTM):
            lstm_module = m
            break
    if lstm_module is None:
        for _, m in model.named_modules():
            if "lstm" in m.__class__.__name__.lower():
                lstm_module = m
                break
    if lstm_module is None:
        raise RuntimeError("Could not find an LSTM-like module to hook.")

    C_list, Xinp_list, date_windows =[], [], []

    def hook_fn(module, inputs, outputs):
        # standard LSTM: (output_seq, (h_n, c_n))
        if isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], tuple):
            output_seq, (h_n, c_n) = outputs
            
            C_list.append(c_n[-1].detach().cpu().numpy())  
        else:
            raise RuntimeError("Hook outputs were not (output_seq, (h_n, c_n)).")

    handle = lstm_module.register_forward_hook(hook_fn)

    with torch.no_grad():
        for batch in loader:
            # store flattened input window (batch, seq_len*n_features)
            x_d = batch["x_d"]
            Xinp_list.append(x_d.numpy().reshape(x_d.shape[0], -1))

            _ = run_forward(model, batch, device)

            # batch["date"] is a list of seq_len arrays
            date_windows.extend(batch["date"])

    handle.remove()

    Xinp = np.concatenate(Xinp_list, axis=0)

    C = np.concatenate(C_list, axis=0)

    # end date = last timestamp in each window
    end_dates = pd.to_datetime([np.array(w).ravel()[-1] for w in date_windows])

    return Xinp, C, end_dates

#MAKE TESTERS (TRAIN / VAL / TEST)
run_dir = Path(run_dir)
cfg = Config(run_dir / "config.yml")

tester_train = get_tester(cfg=cfg, run_dir=run_dir, period="train", init_model=True)

#EXTRACT STATES + INPUTS
Xinp_train,C_train, dates_train = extract_input_cell_and_end_dates(tester_train, basin, batch_size=batch_size)


print("Shapes:")
print("  Input:", Xinp_train.shape)

print("  Cell:", C_train.shape)

print("Date ranges:")
print("  train:", dates_train.min(), "to", dates_train.max())

# ALIGN SOIL MOISTURE

SM_train = sm_series.reindex(dates_train).to_numpy()

print("SM NaNs (train):", np.isnan(SM_train).sum())

def clean_xy(X, y):
    y = np.asarray(y).reshape(-1)
    X = np.asarray(X)
    good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X[good], y[good]

#PROBE 
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

def run_probe(X_train, y_train, name=""):
    Xtr, ytr = clean_xy (X_train, y_train)

    # standardize X using TRAIN stats per feature
    X_mean = Xtr.mean(axis=0)
    X_std  = Xtr.std(axis=0)
    X_std  = np.where(X_std < 1e-8, 1.0, X_std)
    Xtr_s  = (Xtr - X_mean) / X_std

    # standardize y using TRAIN stats
    y_mean = ytr.mean()
    y_std  = ytr.std()
    y_std  = 1.0 if y_std < 1e-8 else y_std
    ytr_n  = (ytr - y_mean) / y_std

    probe = Ridge(alpha = 1.0)
    probe.fit(Xtr_s, ytr_n)

    pred_tr = probe.predict(Xtr_s)


    r2_tr = r2_score(ytr_n, pred_tr)


    print(f"\n{name} PROBE RESULTS")
    print("  R² train:", r2_tr)


# RUN PROBES
run_probe(Xinp_train, SM_train,  name="Input→SM")
run_probe(C_train,    SM_train, name="Cell→SM")



