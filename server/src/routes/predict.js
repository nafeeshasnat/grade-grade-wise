import express from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs/promises';
import { fileURLToPath } from 'url';
import { PrismaClient } from '@prisma/client';
import { authenticateToken } from '../auth.js';
import { runPythonPredict } from '../utils/python-runner.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const router = express.Router();
const prisma = new PrismaClient();

// Configure multer
const storage = multer.diskStorage({
  destination: async (req, file, cb) => {
    const uploadDir = path.join(__dirname, '../../storage/predictions', req.orgId);
    await fs.mkdir(uploadDir, { recursive: true });
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const timestamp = Date.now();
    cb(null, `student_${timestamp}.json`);
  }
});

const upload = multer({ 
  storage,
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'application/json' || file.originalname.endsWith('.json')) {
      cb(null, true);
    } else {
      cb(new Error('Only JSON files are allowed'));
    }
  }
});

// Make prediction
router.post('/', authenticateToken, upload.single('studentFile'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Student file is required' });
    }

    // Get latest succeeded model
    const lastSucceeded = await prisma.modelRun.findFirst({
      where: { 
        orgId: req.orgId,
        status: 'SUCCEEDED'
      },
      orderBy: { createdAt: 'desc' }
    });

    if (!lastSucceeded) {
      return res.status(400).json({ error: 'No trained model found. Please train a model first.' });
    }

    // Parse student data to get ID
    const studentData = JSON.parse(await fs.readFile(req.file.path, 'utf-8'));
    const studentId = studentData.student_id || studentData.id || 'unknown';

    // Prepare output file path
    const outFile = req.file.path.replace('.json', '_prediction.json');

    // Run prediction
    const result = await runPythonPredict(
      req.orgId,
      req.file.path,
      lastSucceeded.artifactsDir,
      outFile
    );

    if (result.status !== 'ok') {
      return res.status(500).json({ error: result.error || 'Prediction failed' });
    }

    // Save prediction record
    const prediction = await prisma.prediction.create({
      data: {
        orgId: req.orgId,
        studentId,
        inputPath: req.file.path,
        outFile,
        results: result.prediction,
        summary: result.prediction
      }
    });

    res.json({
      id: prediction.id,
      studentId: prediction.studentId,
      results: prediction.results,
      createdAt: prediction.createdAt
    });
  } catch (error) {
    console.error('Prediction error:', error);
    res.status(500).json({ error: 'Prediction failed: ' + error.message });
  }
});

// List predictions
router.get('/', authenticateToken, async (req, res) => {
  try {
    const predictions = await prisma.prediction.findMany({
      where: { orgId: req.orgId },
      orderBy: { createdAt: 'desc' },
      take: 50
    });

    res.json(predictions);
  } catch (error) {
    console.error('List predictions error:', error);
    res.status(500).json({ error: 'Failed to list predictions' });
  }
});

// Get single prediction
router.get('/:id', authenticateToken, async (req, res) => {
  try {
    const prediction = await prisma.prediction.findFirst({
      where: { 
        id: req.params.id,
        orgId: req.orgId
      }
    });

    if (!prediction) {
      return res.status(404).json({ error: 'Prediction not found' });
    }

    res.json(prediction);
  } catch (error) {
    console.error('Get prediction error:', error);
    res.status(500).json({ error: 'Failed to get prediction' });
  }
});

export default router;
