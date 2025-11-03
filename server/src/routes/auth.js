import express from 'express';
import bcrypt from 'bcrypt';
import jwt from 'jsonwebtoken';
import { PrismaClient } from '@prisma/client';
import { authenticateToken } from '../auth.js';

const router = express.Router();
const prisma = new PrismaClient();

// Sign up
router.post('/signup', async (req, res) => {
  try {
    const { orgName, email, password } = req.body;

    if (!orgName || !email || !password) {
      return res.status(400).json({ error: 'orgName, email, and password are required' });
    }

    // Check if user exists
    const existingUser = await prisma.user.findUnique({ where: { email } });
    if (existingUser) {
      return res.status(400).json({ error: 'Email already registered' });
    }

    // Hash password
    const passwordHash = await bcrypt.hash(password, 10);

    // Create org and user in transaction
    const result = await prisma.$transaction(async (tx) => {
      const org = await tx.organization.create({
        data: { name: orgName }
      });

      const user = await tx.user.create({
        data: {
          orgId: org.id,
          email,
          passwordHash
        },
        include: { org: true }
      });

      return { org, user };
    });

    // Generate JWT
    const token = jwt.sign(
      { userId: result.user.id, orgId: result.org.id },
      process.env.JWT_SECRET,
      { expiresIn: '7d' }
    );

    res.status(201).json({
      token,
      user: {
        id: result.user.id,
        email: result.user.email,
        createdAt: result.user.createdAt
      },
      org: {
        id: result.org.id,
        name: result.org.name
      }
    });
  } catch (error) {
    console.error('Signup error:', error);
    res.status(500).json({ error: 'Signup failed' });
  }
});

// Sign in
router.post('/signin', async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({ error: 'Email and password are required' });
    }

    // Find user with org
    const user = await prisma.user.findUnique({
      where: { email },
      include: { org: true }
    });

    if (!user) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    // Verify password
    const validPassword = await bcrypt.compare(password, user.passwordHash);
    if (!validPassword) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    // Generate JWT
    const token = jwt.sign(
      { userId: user.id, orgId: user.orgId },
      process.env.JWT_SECRET,
      { expiresIn: '7d' }
    );

    res.json({
      token,
      user: {
        id: user.id,
        email: user.email,
        createdAt: user.createdAt
      },
      org: {
        id: user.org.id,
        name: user.org.name
      }
    });
  } catch (error) {
    console.error('Signin error:', error);
    res.status(500).json({ error: 'Signin failed' });
  }
});

// Get current user
router.get('/me', authenticateToken, async (req, res) => {
  res.json({
    user: {
      id: req.user.id,
      email: req.user.email,
      createdAt: req.user.createdAt
    },
    org: {
      id: req.user.org.id,
      name: req.user.org.name
    }
  });
});

export default router;
