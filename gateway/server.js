const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const http = require('http');

const app = express();
const PORT = process.env.PORT || 8000;
const ENGINE_URL = process.env.ENGINE_URL || 'http://127.0.0.1:8001';

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// CORS middleware
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
  if (req.method === 'OPTIONS') {
    return res.sendStatus(200);
  }
  next();
});

// Configure multer for file uploads with size and validation
const upload = multer({
  limits: {
    fileSize: 25 * 1024 * 1024 // 25MB Max
  },
  fileFilter: (req, file, cb) => {
    const allowedExts = ['.png', '.jpg', '.jpeg', '.webp'];
    const ext = path.extname(file.originalname).toLowerCase();
    if (!allowedExts.includes(ext)) {
      return cb(new Error('Unsupported or corrupt image. Please upload PNG, JPG, or WebP.'));
    }
    cb(null, true);
  }
});

// Proxy and utility helper to talk to python compositing engine
function forwardToEngine(options, postData = null, res) {
  const reqOpts = {
    hostname: '127.0.0.1',
    port: 8001,
    path: options.path,
    method: options.method,
    headers: options.headers || {}
  };

  const req = http.request(reqOpts, (engineRes) => {
    res.status(engineRes.statusCode);
    for (const key in engineRes.headers) {
      res.setHeader(key, engineRes.headers[key]);
    }
    engineRes.pipe(res);
  });

  req.on('error', (err) => {
    res.status(500).json({
      error: {
        code: 'ENGINE_CONNECTION_FAILED',
        message: 'Could not connect to the compositing engine'
      }
    });
  });

  if (postData) {
    req.write(postData);
  }
  req.end();
}

function checkMagicBytes(buffer) {
  const isPng = buffer[0] === 0x89 && buffer[1] === 0x50 && buffer[2] === 0x4E && buffer[3] === 0x47;
  const isJpg = buffer[0] === 0xFF && buffer[1] === 0xD8 && buffer[2] === 0xFF;
  const isWebp = buffer.slice(8, 12).toString() === 'WEBP';
  return isPng || isJpg || isWebp;
}

// ----------------- Gateway Routes -----------------

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', service: 'Node.js Gateway API' });
});

app.post('/api/templates', (req, res) => {
  const category = req.query.category || '';
  forwardToEngine({
    path: `/api/templates?category=${category}`,
    method: 'POST'
  }, null, res);
});

app.get('/api/templates/:id', (req, res) => {
  const tid = req.params.id;
  if (tid.includes('..') || tid.includes('/') || tid.includes('\\')) {
    return res.status(400).json({ error: { code: 'INVALID_PATH', message: 'Invalid template path' } });
  }
  forwardToEngine({
    path: `/api/templates/${tid}`,
    method: 'GET'
  }, null, res);
});

app.get('/api/templates/:id/asset/:file', (req, res) => {
  const tid = req.params.id;
  const file = req.params.file;
  if (tid.includes('..') || file.includes('..') || tid.includes('/') || file.includes('/')) {
    return res.status(400).json({ error: { code: 'INVALID_PATH', message: 'Invalid asset path' } });
  }
  forwardToEngine({
    path: `/api/templates/${tid}/asset/${file}`,
    method: 'GET'
  }, null, res);
});

app.post('/api/designs/upload', upload.single('file'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: { code: 'MISSING_FILE', message: 'No file uploaded' } });
  }

  if (!checkMagicBytes(req.file.buffer)) {
    return res.status(400).json({ error: { code: 'INVALID_SIGNATURE', message: 'MIME type signature check failed.' } });
  }

  const boundary = '----WebKitFormBoundary' + Math.random().toString(36).substring(2);
  let postData = [];
  postData.push(Buffer.from(`--${boundary}\r\n`));
  postData.push(Buffer.from(`Content-Disposition: form-data; name="file"; filename="${req.file.originalname}"\r\n`));
  postData.push(Buffer.from(`Content-Type: ${req.file.mimetype}\r\n\r\n`));
  postData.push(req.file.buffer);
  postData.push(Buffer.from(`\r\n--${boundary}--\r\n`));

  const totalBuffer = Buffer.concat(postData);

  forwardToEngine({
    path: '/api/designs/upload',
    method: 'POST',
    headers: {
      'Content-Type': `multipart/form-data; boundary=${boundary}`,
      'Content-Length': totalBuffer.length
    }
  }, totalBuffer, res);
});

app.post('/api/designs/:id/remove-bg', (req, res) => {
  const id = req.params.id;
  forwardToEngine({
    path: `/api/designs/${id}/remove-bg`,
    method: 'POST'
  }, null, res);
});

app.get('/api/designs/:id/file', (req, res) => {
  const id = req.params.id;
  forwardToEngine({
    path: `/api/designs/${id}/file`,
    method: 'GET'
  }, null, res);
});

app.post('/api/render', (req, res) => {
  const postBody = JSON.stringify(req.body);
  forwardToEngine({
    path: '/api/render',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(postBody)
    }
  }, Buffer.from(postBody), res);
});

app.get('/api/render/:job_id/download', (req, res) => {
  const jid = req.params.job_id;
  forwardToEngine({
    path: `/api/render/${jid}/download`,
    method: 'GET'
  }, null, res);
});

app.get('/api/history', (req, res) => {
  forwardToEngine({
    path: '/api/history',
    method: 'GET'
  }, null, res);
});

app.delete('/api/history/:job_id', (req, res) => {
  const jid = req.params.job_id;
  forwardToEngine({
    path: `/api/history/${jid}`,
    method: 'DELETE'
  }, null, res);
});

app.post('/api/templates/ingest', upload.single('file'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: { code: 'MISSING_FILE', message: 'No file uploaded' } });
  }

  const boundary = '----WebKitFormBoundary' + Math.random().toString(36).substring(2);
  let postData = [];
  postData.push(Buffer.from(`--${boundary}\r\n`));
  postData.push(Buffer.from(`Content-Disposition: form-data; name="file"; filename="${req.file.originalname}"\r\n`));
  postData.push(Buffer.from(`Content-Type: ${req.file.mimetype}\r\n\r\n`));
  postData.push(req.file.buffer);
  postData.push(Buffer.from(`\r\n`));

  const fields = ['id', 'category', 'subtype', 'label', 'fold_intensity'];
  for (const f of fields) {
    if (req.body[f]) {
      postData.push(Buffer.from(`--${boundary}\r\n`));
      postData.push(Buffer.from(`Content-Disposition: form-data; name="${f}"\r\n\r\n`));
      postData.push(Buffer.from(`${req.body[f]}\r\n`));
    }
  }
  postData.push(Buffer.from(`--${boundary}--\r\n`));

  const totalBuffer = Buffer.concat(postData);

  forwardToEngine({
    path: '/api/templates/ingest',
    method: 'POST',
    headers: {
      'Content-Type': `multipart/form-data; boundary=${boundary}`,
      'Content-Length': totalBuffer.length
    }
  }, totalBuffer, res);
});

app.use(express.static(path.join(__dirname, '../frontend')));

app.get('*', (req, res, next) => {
  if (req.path.startsWith('/api')) {
    return next();
  }
  res.sendFile(path.join(__dirname, '../frontend/index.html'));
});

app.listen(PORT, () => {
  console.log(`Node.js Gateway service running on port ${PORT}`);
});
