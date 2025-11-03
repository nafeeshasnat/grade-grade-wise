import { spawn } from 'child_process';
import fs from 'fs/promises';
import path from 'path';

const PYTHON_BIN = process.env.PYTHON_BIN || 'python3';

export async function runPythonTrain(orgId, runId, trainJsonPath, configJsonPath, outDir, prisma) {
  const logPath = path.join(outDir, 'train.log');
  const logStream = await fs.open(logPath, 'w');

  const args = [
    'ml/train.py',
    '--org-id', orgId,
    '--train-json', trainJsonPath,
    '--config-json', configJsonPath,
    '--out-dir', outDir
  ];

  const pythonProcess = spawn(PYTHON_BIN, args);

  let resultJson = null;

  pythonProcess.stdout.on('data', async (data) => {
    const text = data.toString();
    await logStream.write(text);

    // Look for __RESULT__ line
    const lines = text.split('\n');
    for (const line of lines) {
      if (line.startsWith('__RESULT__')) {
        const jsonStr = line.substring('__RESULT__'.length);
        try {
          resultJson = JSON.parse(jsonStr);
        } catch (error) {
          console.error('Failed to parse result JSON:', error);
        }
      }
    }
  });

  pythonProcess.stderr.on('data', async (data) => {
    await logStream.write(data.toString());
  });

  return new Promise((resolve, reject) => {
    pythonProcess.on('close', async (code) => {
      await logStream.close();

      if (resultJson && resultJson.status === 'ok') {
        // Update model run with success
        await prisma.modelRun.update({
          where: { id: runId },
          data: {
            status: 'SUCCEEDED',
            metrics: resultJson.metrics || {},
            plots: resultJson.plots || {},
            finishedAt: new Date()
          }
        });
        resolve(resultJson);
      } else {
        // Update model run with failure
        await prisma.modelRun.update({
          where: { id: runId },
          data: {
            status: 'FAILED',
            finishedAt: new Date()
          }
        });
        reject(new Error(resultJson?.error || `Training failed with code ${code}`));
      }
    });

    pythonProcess.on('error', async (error) => {
      await logStream.close();
      await prisma.modelRun.update({
        where: { id: runId },
        data: {
          status: 'FAILED',
          finishedAt: new Date()
        }
      });
      reject(error);
    });
  });
}

export async function runPythonPredict(orgId, studentJsonPath, artifactsDir, outFile) {
  const args = [
    'ml/predict.py',
    '--org-id', orgId,
    '--student-json', studentJsonPath,
    '--artifacts-dir', artifactsDir,
    '--out-file', outFile
  ];

  const pythonProcess = spawn(PYTHON_BIN, args);

  let stdout = '';
  let stderr = '';
  let resultJson = null;

  pythonProcess.stdout.on('data', (data) => {
    const text = data.toString();
    stdout += text;

    // Look for __RESULT__ line
    const lines = text.split('\n');
    for (const line of lines) {
      if (line.startsWith('__RESULT__')) {
        const jsonStr = line.substring('__RESULT__'.length);
        try {
          resultJson = JSON.parse(jsonStr);
        } catch (error) {
          console.error('Failed to parse result JSON:', error);
        }
      }
    }
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
