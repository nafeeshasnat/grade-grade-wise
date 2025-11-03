// Validate and clamp training configuration

const DEFAULTS = {
  RANDOM_SEED: 42,
  TEST_SIZE: 0.2,
  THREADS: 4,
  RF_TREES: 300,
  LGBM_N_ESTIMATORS: 2000,
  MLP_HIDDEN: 64,
  MLP_EPOCHS: 300,
  MLP_PATIENCE: 40,
  SVR_ENABLE: true,
  RISK_HIGH_MAX: 3.30,
  RISK_MED_MAX: 3.50,
  GRADE_POINTS: {
    "A+": 4.0,
    "A": 3.75,
    "A-": 3.5,
    "B+": 3.25,
    "B": 3.0,
    "B-": 2.75,
    "C+": 2.5,
    "C": 2.25,
    "D": 2.0,
    "F": 0.0
  }
};

const BOUNDS = {
  TEST_SIZE: [0.1, 0.3],
  THREADS: [1, 16],
  RF_TREES: [50, 1000],
  LGBM_N_ESTIMATORS: [200, 4000],
  MLP_HIDDEN: [16, 256],
  MLP_EPOCHS: [50, 600],
  MLP_PATIENCE: [10, 100]
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export function validateConfig(inputConfig) {
  const config = { ...DEFAULTS };

  // Apply user values with clamping
  if (inputConfig.RANDOM_SEED !== undefined) {
    config.RANDOM_SEED = Math.floor(inputConfig.RANDOM_SEED);
  }

  if (inputConfig.TEST_SIZE !== undefined) {
    config.TEST_SIZE = clamp(inputConfig.TEST_SIZE, ...BOUNDS.TEST_SIZE);
  }

  if (inputConfig.THREADS !== undefined) {
    config.THREADS = Math.floor(clamp(inputConfig.THREADS, ...BOUNDS.THREADS));
  }

  if (inputConfig.RF_TREES !== undefined) {
    config.RF_TREES = Math.floor(clamp(inputConfig.RF_TREES, ...BOUNDS.RF_TREES));
  }

  if (inputConfig.LGBM_N_ESTIMATORS !== undefined) {
    config.LGBM_N_ESTIMATORS = Math.floor(clamp(inputConfig.LGBM_N_ESTIMATORS, ...BOUNDS.LGBM_N_ESTIMATORS));
  }

  if (inputConfig.MLP_HIDDEN !== undefined) {
    config.MLP_HIDDEN = Math.floor(clamp(inputConfig.MLP_HIDDEN, ...BOUNDS.MLP_HIDDEN));
  }

  if (inputConfig.MLP_EPOCHS !== undefined) {
    config.MLP_EPOCHS = Math.floor(clamp(inputConfig.MLP_EPOCHS, ...BOUNDS.MLP_EPOCHS));
  }

  if (inputConfig.MLP_PATIENCE !== undefined) {
    config.MLP_PATIENCE = Math.floor(clamp(inputConfig.MLP_PATIENCE, ...BOUNDS.MLP_PATIENCE));
  }

  if (inputConfig.SVR_ENABLE !== undefined) {
    config.SVR_ENABLE = Boolean(inputConfig.SVR_ENABLE);
  }

  if (inputConfig.RISK_HIGH_MAX !== undefined) {
    config.RISK_HIGH_MAX = parseFloat(inputConfig.RISK_HIGH_MAX);
  }

  if (inputConfig.RISK_MED_MAX !== undefined) {
    config.RISK_MED_MAX = parseFloat(inputConfig.RISK_MED_MAX);
  }

  // Validate RISK thresholds
  if (config.RISK_HIGH_MAX > config.RISK_MED_MAX) {
    throw new Error('RISK_HIGH_MAX must be <= RISK_MED_MAX');
  }

  // Validate GRADE_POINTS
  if (inputConfig.GRADE_POINTS) {
    const gradePoints = {};
    for (const [key, value] of Object.entries(inputConfig.GRADE_POINTS)) {
      const numValue = parseFloat(value);
      if (isNaN(numValue) || numValue < 0 || numValue > 10) {
        throw new Error(`Invalid grade point value for ${key}: must be between 0 and 10`);
      }
      gradePoints[key] = numValue;
    }
    config.GRADE_POINTS = gradePoints;
  }

  return config;
}
