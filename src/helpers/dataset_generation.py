from pathlib import Path

import numpy as np
import pandas as pd

HEART_DISEASE_PATH = Path(__file__).resolve().parents[2] / "data" / "heart_disease.csv"
HEART_DISEASE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
# Reduced 8-variable version: keeps predictors informative for `num`, dropping the
# weaker/noisier ones (age, trestbps, fbs, restecg, slope, ca). sex and chol were swapped in
# (replacing age and ca) to test explicit dependency hypotheses on those two specifically. Also
# cuts the worst-case joint-mixture blowup that OOM-killed the full 14-variable run from 3^14
# (~4.8M) to 3^8 (~6.6k) -- see estimate_joint_components in llm_synthesis.py.
HEART_DISEASE_SMALL_COLUMNS = ["sex", "cp", "chol", "thalach", "exang", "oldpeak", "thal", "num"]

# RefineStat's own 5 native benchmarks (commons/data_pymc.py's *_template arrays, copied
# verbatim) -- small, fixed reference datasets rather than a parametric simulator, so (like
# heart_disease) get_dataset shuffles-without-replacement / bootstraps from these fixed pools
# instead of drawing fresh synthetic samples.
EIGHT_SCHOOLS_COLUMNS = ["y", "sigma"]
EIGHT_SCHOOLS_DATA = {
    "y": [28, 8, -3, 7, -1, 1, 18, 12],
    "sigma": [15, 10, 16, 11, 9, 11, 10, 18],
}

DUGONGS_COLUMNS = ["X", "y"]
DUGONGS_DATA = {
    "X": [1, 1.5, 1.5, 1.5, 2.5, 4, 5, 5, 7, 8, 8.5, 9, 9.5, 9.5, 10, 12, 12, 13, 13, 14.5,
          15.5, 15.5, 16.5, 17, 22.5, 29, 31.5],
    "y": [1.8, 1.85, 1.87, 1.77, 2.02, 2.27, 2.15, 2.26, 2.47, 2.19, 2.26, 2.4, 2.39, 2.41,
          2.5, 2.32, 2.32, 2.43, 2.47, 2.56, 2.65, 2.47, 2.64, 2.56, 2.7, 2.72, 2.57],
}

GP_COLUMNS = ["x", "y", "k"]
GP_DATA = {
    "x": [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10],
    "y": [4.75906, 1.59423, 2.99548, 5.27501, 1.66472, 2.24347, 2.8914, 4.08681, 4.60588,
          0.802364, 3.92136],
    "k": [40, 37, 29, 12, 4, 3, 9, 19, 77, 82, 33],
}

GLM_COLUMNS = ["year", "C", "N"]
GLM_DATA = {
    "year": [-0.95, -0.9, -0.85, -0.8, -0.75, -0.7, -0.65, -0.6, -0.55, -0.5, -0.45, -0.4,
             -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
             0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
    "C": [27, 42, 35, 55, 61, 19, 41, 74, 43, 42, 73, 37, 48, 49, 19, 72, 30, 18, 31, 71, 63,
          51, 48, 73, 49, 54, 43, 59, 30, 24, 62, 55, 51, 47, 14, 27, 45, 20, 26, 19],
    "N": [43, 83, 53, 91, 95, 24, 62, 91, 64, 57, 97, 56, 74, 66, 28, 92, 40, 23, 46, 96, 91,
          75, 71, 100, 72, 77, 64, 68, 43, 32, 97, 92, 75, 84, 22, 58, 81, 37, 45, 39],
}

SURGICAL_COLUMNS = ["n", "r"]
SURGICAL_DATA = {
    "n": [47, 148, 119, 810, 211, 196, 148, 215, 207, 97, 256, 360],
    "r": [0, 18, 8, 46, 8, 13, 9, 31, 14, 8, 29, 24],
}


def generate_fixed_dataset(data_size, columns_dict, columns):
    """Shuffle-without-replacement / bootstrap from a small fixed reference dataset -- same
    convention as generate_heart_disease_dataset, for datasets that are real/reference data
    rather than a parametric simulator (see module-level RefineStat benchmark comment)."""
    arr = np.array([columns_dict[c] for c in columns], dtype=float).T
    n = len(arr)
    if data_size <= n:
        idx = np.random.permutation(n)[:data_size]
    else:
        idx = np.random.randint(0, n, size=data_size)
    return arr[idx].tolist()


def get_dataset(program: str, data_size: int):
    if program == 'if':
        data = generate_if_dataset(data_size)
    elif program == 'mog1':
        data = generate_mog1_dataset(data_size)
    elif program == 'burglary':
        data = generate_burglary_dataset(data_size)
    elif program == 'csi':
        data = generate_csi_dataset(data_size)
    elif program == 'easytugwar':
        data = generate_easytugwar_dataset(data_size)
    elif program == 'biasedtugwar':
        data = generate_biasedtugwar_dataset(data_size)
    elif program == 'mixedcondition':
        data = generate_mixedcondition_dataset(data_size)
    elif program == 'multiplebranches':
        data = generate_multiplebranches_dataset(data_size)
    elif program == 'eyecolor':
        data = generate_eyecolor_dataset(data_size)
    elif program == 'hurricane':
        data = generate_hurricane_dataset(data_size)
    elif program == 'heart_disease':
        data = generate_heart_disease_dataset(data_size)
    elif program == 'heart_disease_small':
        data = generate_heart_disease_dataset(data_size, columns=HEART_DISEASE_SMALL_COLUMNS)
    elif program == 'eight_schools':
        data = generate_fixed_dataset(data_size, EIGHT_SCHOOLS_DATA, EIGHT_SCHOOLS_COLUMNS)
    elif program == 'dugongs':
        data = generate_fixed_dataset(data_size, DUGONGS_DATA, DUGONGS_COLUMNS)
    elif program == 'gp':
        data = generate_fixed_dataset(data_size, GP_DATA, GP_COLUMNS)
    elif program == 'glm':
        data = generate_fixed_dataset(data_size, GLM_DATA, GLM_COLUMNS)
    elif program == 'surgical':
        data = generate_fixed_dataset(data_size, SURGICAL_DATA, SURGICAL_COLUMNS)
    else:
        raise ValueError(f"Unknown program type: {program}")
    return data

def get_var_names(program: str):
    if program == 'if':
        return ['a', 'b']
    elif program == 'mog1':
        return ['mu', 'sigma', 'x']
    elif program == 'burglary':
        return ['burglary', 'earthquake', 'alarm', 'johncalls']
    elif program == 'csi':
        return ['u', 'v', 'w', 'x']
    elif program == 'easytugwar':
        return ['skill1', 'skill2', 'p1wins']
    elif program == 'biasedtugwar':
        return ['skill1', 'skill2', 'p1wins']
    elif program == 'mixedcondition':
        return ['u', 'v', 'w']
    elif program == 'multiplebranches':
        return ['contentDifficulty', 'questionsAfterLectureLength']
    elif program == 'eyecolor':
        return ['eyecolor', 'haircolor', 'hairlenght']
    elif program == 'hurricane':
        return ['preplevel', 'damage']
    elif program == 'heart_disease':
        return HEART_DISEASE_COLUMNS
    elif program == 'heart_disease_small':
        return HEART_DISEASE_SMALL_COLUMNS
    elif program == 'eight_schools':
        return EIGHT_SCHOOLS_COLUMNS
    elif program == 'dugongs':
        return DUGONGS_COLUMNS
    elif program == 'gp':
        return GP_COLUMNS
    elif program == 'glm':
        return GLM_COLUMNS
    elif program == 'surgical':
        return SURGICAL_COLUMNS
    else:
        raise ValueError(f"Unknown program type: {program}")



def generate_if_dataset(data_size):
    data = []
    for _ in range(data_size):
        a = np.random.normal(1, 2)
        if a < 0:
            b = a * 3 + np.random.normal(0, 1)
        else:
            b = np.random.normal(8, 1)
        data.append([a, b])
    return data

def generate_mog1_dataset(data_size):
    data = []
    for _ in range(data_size):
        mu = np.random.normal(20, 3)
        sigma = np.random.normal(2, 1)
        x = mu + sigma * np.random.normal(1, 1)
        data.append([mu, sigma, x])
    return data

def generate_burglary_dataset(data_size):
    data = []
    for _ in range(data_size):
        burglary = np.random.binomial(1, 0.001)  # 0.001
        earthquake = np.random.binomial(1, 0.002) # 0.002
        if burglary:
            if earthquake:
                alarm = np.random.binomial(1, 0.95)
            else:
                alarm = np.random.binomial(1, 0.94)
        else:
            if earthquake:
                alarm = np.random.binomial(1, 0.29)
            else:
                alarm = np.random.binomial(1, 0.001) # 0.001
        if alarm:
            johncalls = np.random.binomial(1, 0.9)
        else:
            johncalls = np.random.binomial(1, 0.05)
        data.append([burglary, earthquake, alarm, johncalls])
    return data

def generate_csi_dataset(data_size):
    data = []
    for _ in range(data_size):
        u = np.random.binomial(1, 0.3)
        v = np.random.binomial(1, 0.9)
        w = np.random.binomial(1, 0.1)
        if u:
            if w:
                x = np.random.binomial(1, 0.8)
            else:
                x = np.random.binomial(1, 0.2)
        else:
            if v:
                x = np.random.binomial(1, 0.8)
            else:
                x = np.random.binomial(1, 0.2)
        data.append([u, v, w, x])
    return data

def generate_easytugwar_dataset(data_size):
    data = []
    for _ in range(data_size):
        skill1 = np.random.normal(20, 4)
        skill2 = np.random.normal(20, 4)
        if skill1 > skill2:
            p1wins = 1.0
        else:
            p1wins = 0.0
        data.append([skill1, skill2, p1wins])
    return data

def generate_biasedtugwar_dataset(data_size):
    data = []
    for _ in range(data_size):
        skill1 = np.random.normal(20, 4)
        skill2 = np.random.normal(20, 4)
        if 1.3 * skill2 - skill1 < 0:
            p1wins = 1.0
        else:
            p1wins = 0.0
        data.append([skill1, skill2, p1wins])
    return data

def generate_mixedcondition_dataset(data_size):
    data = []
    for _ in range(data_size):
        u = np.random.binomial(1, 0.3)
        v = np.random.normal(10, 2)
        if u == 1 and v > 12:
            w = np.random.normal(12, 2)
        else:
            w = np.random.normal(6, 2)
        data.append([u, v, w])
    return data

def generate_multiplebranches_dataset(data_size):
    data = []
    for _ in range(data_size):
        contentDifficulty = np.random.normal(30, 5)
        if contentDifficulty < 35:
            if contentDifficulty < 20:
                questionsAfterLectureLength = np.random.normal(2, 1)
            else:
                questionsAfterLectureLength = np.random.normal(10, 3)
        else:
            questionsAfterLectureLength = np.random.normal(25, 6)
        data.append([contentDifficulty, questionsAfterLectureLength])
    return data

def generate_eyecolor_dataset(data_size):
    # real_programs.py's eyecolor weights [0.8, 0.05, 0.04, 0.01] sum to 0.90, not 1
    # (already present in the ported program, inherited from the upstream .soga baseline
    # as-is) -- normalized here since np.random.choice requires a valid distribution.
    eyecolor_p = np.array([0.8, 0.05, 0.04, 0.01])
    eyecolor_p = eyecolor_p / eyecolor_p.sum()
    haircolor_p_by_eyecolor = {
        0: [0.8, 0.05, 0.04, 0.01, 0.1],
        1: [0.7, 0.15, 0.04, 0.01, 0.1],
        2: [0.4, 0.3, 0.18, 0.02, 0.1],
        3: [0.4, 0.29, 0.18, 0.03, 0.1],
    }
    data = []
    for _ in range(data_size):
        eyecolor = np.random.choice([0, 1, 2, 3], p=eyecolor_p)
        haircolor = np.random.choice([0, 1, 2, 3, 4], p=haircolor_p_by_eyecolor[eyecolor])
        hairlenght = np.random.choice([0, 1, 2], p=[0.6, 0.15, 0.25])
        data.append([eyecolor, haircolor, hairlenght])
    return data

def generate_heart_disease_dataset(data_size, columns=HEART_DISEASE_COLUMNS):
    # Real UCI dataset (archive.ics.uci.edu/dataset/45/heart-disease), 297 rows after dropping
    # the 6 rows with missing `ca`/`thal` values -- unlike every other benchmark here, this is
    # not a forward sampler from a known generative process, it's real-world data. With only 297
    # rows available, data_size is expected to be <= 297 (shuffled, sampled without replacement,
    # so every row returned is a genuine distinct real observation -- e.g. an 80/20 split via
    # ALPS_HELD_OUT_SELECTION/ALPS_TRAIN_FRAC on data_size=297). Falls back to bootstrap
    # resampling (with replacement) only if a larger data_size is ever requested.
    df = pd.read_csv(HEART_DISEASE_PATH)
    if data_size <= len(df):
        idx = np.random.permutation(len(df))[:data_size]
    else:
        idx = np.random.randint(0, len(df), size=data_size)
    return df.iloc[idx][columns].values.tolist()

def generate_hurricane_dataset(data_size):
    damage_p_by_preplevel = {0: [0.2, 0.8], 1: [0.2, 0.8], 2: [0.8, 0.2]}
    data = []
    for _ in range(data_size):
        preplevel = np.random.choice([0, 1, 2], p=[0.5, 0.2, 0.3])
        damage = np.random.choice([0, 1], p=damage_p_by_preplevel[preplevel])
        data.append([preplevel, damage])
    return data