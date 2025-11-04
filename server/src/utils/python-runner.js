import { spawn, spawnSync } from 'child_process';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
function resolvePythonBinary() {
  if (process.env.PYTHON_BIN) {
    return process.env.PYTHON_BIN;
  }

  const candidates = process.platform === 'win32'
    ? ['python', 'py', 'python3']
    : ['python3', 'python'];

  for (const candidate of candidates) {
    const result = spawnSync(candidate, ['--version'], { stdio: 'ignore' });
    if (!result.error && result.status === 0) {
      return candidate;
    }
  }

  // Fall back to the first candidate; spawn will surface a clearer error later
  return candidates[0];
}

const PYTHON_BIN = resolvePythonBinary();

// Resolve script paths ABSOLUTELY so CWD doesn't matter
const TRAIN_SCRIPT = path.resolve(__dirname, '../../ml/train.py');
const PREDICT_SCRIPT = path.resolve(__dirname, '../../ml/predict.py');

// Robust line-buffered capture that tolerates chunk splits
function makeResultCatcher(onResult) {
  let buf = '';
  return (chunk) => {
    buf += chunk;
    // handle Windows \r\n and partial frames
    let idx;
    while ((idx = buf.indexOf('__RESULT__')) !== -1) {
      // try to extract JSON object after marker
      const after = buf.slice(idx + '__RESULT__'.length);
      // Find end of JSON by last '}' before a newline OR try greedy parse
      // Simple strategy: split lines and try parse the first line after the marker
      const nl = after.indexOf('\n');
      const candidate = (nl === -1 ? after : after.slice(0, nl)).trim();
      try {
        const parsed = JSON.parse(candidate);
        onResult(parsed);
        // consume up to end of that line
        buf = (nl === -1 ? '' : after.slice(nl + 1));
      } catch {
        // Not a full JSON yet -> wait for more data
        break;
      }
    }
    // Keep buffer from growing unbounded
    if (buf.length > 1_000_000) buf = buf.slice(-100_000);
  };
}

export async function runPythonTrain(orgId, runId, trainJsonPath, configJsonPath, outDir, prisma) {
  const logPath = path.join(outDir, 'train.log');
  const logHandle = await fs.open(logPath, 'a'); // append; create if missing

  // ALSO keep a copy in config.json under outDir (some UIs expect it there)
  try {
    await fs.copyFile(configJsonPath, path.join(outDir, 'config.json'));
  } catch { } // best effort

  const args = [
    TRAIN_SCRIPT,
    '--org-id', orgId,
    '--train-json', trainJsonPath,
    '--config-json', configJsonPath,
    '--out-dir', outDir,
  ];

  // Ensure unbuffered output and inherit env
  const pythonProcess = spawn(PYTHON_BIN, args, {
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  let resultJson = null;
  const catchResult = makeResultCatcher((r) => { resultJson = r; });

  pythonProcess.stdout.on('data', async (data) => {
    const text = data.toString();
    catchResult(text);
    await logHandle.write(text);
  });

  pythonProcess.stderr.on('data', async (data) => {
    await logHandle.write(data.toString());
  });

  return new Promise((resolve, reject) => {
    pythonProcess.on('close', async (code) => {
      if (!resultJson || resultJson.status !== 'ok') {
        try {
          await logHandle.write(`Training process exited with code ${code}.\n`);
        } catch (writeErr) {
          console.error('Failed to write close error to log:', writeErr);
        }
      }
      await logHandle.close();

      // Default-safe payloads for Prisma JSON columns
      const metrics = (resultJson && resultJson.metrics) ? resultJson.metrics : {};
      const plots = (resultJson && resultJson.plots) ? resultJson.plots : [];

      if (resultJson && resultJson.status === 'ok') {
        try {
          await prisma.modelRun.update({
            where: { id: runId },
            data: {
              status: 'SUCCEEDED',
              metrics,
              plots,
              finishedAt: new Date(),
            },
          });
        } catch (e) {
          // Don't hide success from caller if DB write fails
          console.error('Prisma update (SUCCEEDED) failed:', e);
        }
        resolve(resultJson);
      } else {
        try {
          await prisma.modelRun.update({
            where: { id: runId },
            data: {
              status: 'FAILED',
              finishedAt: new Date(),
            },
          });
        } catch (e) {
          console.error('Prisma update (FAILED) failed:', e);
        }

        const errMsg = resultJson?.error || `Training failed with code ${code}`;
        reject(new Error(errMsg));
      }
    });

    pythonProcess.on('error', async (error) => {
      try {
        await logHandle.write(`Failed to start Python process (${PYTHON_BIN}): ${error.message}\n`);
      } catch (writeErr) {
        console.error('Failed to write spawn error to log:', writeErr);
      }
      await logHandle.close();
      try {
        await prisma.modelRun.update({
          where: { id: runId },
          data: {
            status: 'FAILED',
            finishedAt: new Date(),
          },
        });
      } catch (e) {
        console.error('Prisma update (FAILED on spawn error) failed:', e);
      }
      reject(error);
    });
  });
}

export async function runPythonPredict(orgId, studentJsonPath, artifactsDir, outFile) {
  const args = [
    PREDICT_SCRIPT,
    '--org-id', orgId,
    '--student-json', studentJsonPath,
    '--artifacts-dir', artifactsDir,
    '--out-file', outFile,
  ];

  const pythonProcess = spawn(PYTHON_BIN, args, {
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  let resultJson = null;
  const catchResult = makeResultCatcher((r) => { resultJson = r; });

  let stderr = '';

  pythonProcess.stdout.on('data', (data) => {
    const text = data.toString();
    catchResult(text);
  });

  pythonProcess.stderr.on('data', (data) => {
    stderr += data.toString();
  });

  return new Promise((resolve, reject) => {
    pythonProcess.on('close', (code) => {
      if (resultJson && resultJson.status === 'ok') {
        resolve(resultJson);
      } else {
        reject(new Error(resultJson?.error || `Prediction failed with code ${code}\n${stderr}`));
      }
    });

    pythonProcess.on('error', (error) => {
      reject(error);
    });
  });
}
