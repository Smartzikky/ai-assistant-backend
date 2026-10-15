require('dotenv').config();
const { HfInference } = require('@huggingface/inference');

(async () => {
  const hf = new HfInference(process.env.HF_API_TOKEN);
  try {
    const r = await hf.chatCompletion({
      model: 'deepseek-ai/DeepSeek-V3',
      provider: 'together',
      messages: [{ role: 'user', content: 'Say hello' }],
    });
    console.log(r.choices[0].message.content);
  } catch (e) {
    console.error(e.message, e.httpResponse?.body || '');
  }
})();
