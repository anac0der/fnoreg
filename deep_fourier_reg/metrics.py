import numpy as np

def rTRE(source_landmarks, target_landmarks, image_diagonal):
    length = range(min(source_landmarks.shape[0], target_landmarks.shape[0]))
    tre = np.sqrt(np.square(source_landmarks[length, 0] - target_landmarks[length, 0]) + np.square(source_landmarks[length, 1] - target_landmarks[length, 1]))
    rtre = tre / image_diagonal
    return rtre

def robustness(fixed_landmarks, moving_landmarks, moved_landmarks, image_diagonal):
    rtre = rTRE(fixed_landmarks, moved_landmarks, image_diagonal)
    rire = rTRE(fixed_landmarks, moving_landmarks, image_diagonal)
    return np.sum(rtre < rire) / fixed_landmarks.shape[0]