import os
import shutil
import pickle

def try_load_pickle(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f), None
    except Exception as e:
        return None, e

def try_load_joblib(path):
    try:
        from joblib import load
        return load(path), None
    except Exception as e:
        return None, e

def try_load_dill(path):
    try:
        import dill
        with open(path, 'rb') as f:
            return dill.load(f), None
    except Exception as e:
        return None, e

def load_with_fallback(path):
    loaders = [try_load_pickle, try_load_joblib, try_load_dill]
    last_err = None
    for loader in loaders:
        res, err = loader(path)
        if res is not None:
            return res
        last_err = err
    raise RuntimeError(f"Failed to load {path}: {last_err}")

def backup(path):
    if os.path.exists(path):
        bak = path + '.bak'
        print(f"Backing up {path} -> {bak}")
        shutil.copy2(path, bak)

def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(repo_root, 'models')
    os.makedirs(models_dir, exist_ok=True)

    files = {
        'model': os.path.join(models_dir, 'adaboost_model.pkl'),
        'preprocessor': os.path.join(models_dir, 'preprocessor.pkl'),
        'feature_info': os.path.join(models_dir, 'feature_info.pkl'),
    }

    # Attempt to load each artifact
    loaded = {}
    for name, path in files.items():
        if not os.path.exists(path):
            print(f"Missing {name} file at {path}; skipping.")
            continue
        print(f"Loading {name} from {path} ...")
        try:
            obj = load_with_fallback(path)
            loaded[name] = obj
            print(f"Loaded {name} successfully.")
        except Exception as e:
            print(f"ERROR: could not load {name}: {e}")

    if not loaded:
        print("No artifacts loaded; aborting re-save.")
        return

    # Re-save using joblib (preferred for sklearn objects)
    try:
        from joblib import dump
    except Exception:
        raise RuntimeError("joblib is required to re-save artifacts. Install it with `pip install joblib`.")

    for name, obj in loaded.items():
        path = files[name]
        backup(path)
        print(f"Re-saving {name} to {path} using joblib.dump...")
        try:
            dump(obj, path)
            print(f"Re-saved {name} to {path}.")
        except Exception as e:
            print(f"Failed to re-save {name}: {e}")

if __name__ == '__main__':
    main()
