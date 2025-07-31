require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');
const { HfInference } = require('@huggingface/inference');
const { OpenAI } = require('openai');

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'client/build')));

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'client/build/index.html'));
});

// Instantiate clients
const useHF = !!process.env.HF_API_TOKEN;
const hf = new HfInference(process.env.HF_API_TOKEN);
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

/**
 * callGPT: Unified function for HF DeepSeek or OpenAI
 * @param {Array} messages - array of { role, content }
 * @returns {Promise<string>}
 */
async function callGPT(messages) {
  if (useHF) {
    try {
      const response = await hf.chatCompletion({
        model: 'deepseek-ai/DeepSeek-V3',
        messages
      });
      return response.choices[0].message.content;
    } catch (err) {
      console.error('Hugging Face error:', err);
      throw new Error(`HF Error: ${err.message}`);
    }
  } else {
    try {
      const response = await openai.chat.completions.create({
        model: 'gpt-3.5-turbo',
        messages
      });
      return response.choices[0].message.content;
    } catch (err) {
      console.error('OpenAI error:', err);
      throw new Error(`OpenAI Error: ${err.message}`);
    }
  }
}

// === ROUTES ===

app.post('/api/health', async (req, res) => {
  const { symptoms } = req.body;
  try {
    const answer = await callGPT([
      { role: 'system', content: 'You are a medical assistant. Provide basic triage and advice with disclaimer.' },
      { role: 'user', content: `Patient symptoms: ${symptoms}` }
    ]);
    res.json({ answer });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/agriculture', async (req, res) => {
  const { context } = req.body;
  try {
    const answer = await callGPT([
      { role: 'system', content: 'You are an agriculture expert. Provide planting and pest control tips.' },
      { role: 'user', content: `Context: ${context}` }
    ]);
    res.json({ answer });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/finance', async (req, res) => {
  const { budgetDetails } = req.body;
  try {
    const answer = await callGPT([
      { role: 'system', content: 'You are a financial advisor. Suggest plans and cost-saving tips.' },
      { role: 'user', content: `Budget details: ${budgetDetails}` }
    ]);
    res.json({ answer });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/general', async (req, res) => {
  const { message } = req.body;
  try {
    const answer = await callGPT([
      { role: 'system', content: 'You are a helpful AI assistant.' },
      { role: 'user', content: message }
    ]);
    res.json({ answer });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Start server
const PORT = process.env.PORT || 4000;
app.listen(PORT, () => console.log(`✅ Backend running on port ${PORT}`));
