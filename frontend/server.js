const express = require('express');
const compression = require('compression');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();
const port = process.env.PORT || 3000;
const proxyTarget =
  process.env.API_PROXY_TARGET ||
  process.env.REACT_APP_API_URL ||
  'http://localhost:8000';

const staticDir = path.join(__dirname, 'build');

app.use((req, res, next) => {
  console.log(`[frontend] ${req.method} ${req.originalUrl}`);
  next();
});

app.use(compression());
app.use(express.static(staticDir, { maxAge: '1h', index: false }));

const proxyConfigs = [
  '/recommend',
  '/user',
  '/availability',
  '/health',
];

proxyConfigs.forEach((basePath) => {
  app.use(
    basePath,
    createProxyMiddleware({
      target: proxyTarget,
      changeOrigin: true,
      logLevel: 'info',
      pathRewrite: (_, req) => req.originalUrl,
    })
  );
});

app.get('*', (req, res) => {
  res.sendFile(path.join(staticDir, 'index.html'));
});

app.listen(port, () => {
  console.log(`Frontend server listening on port ${port}. Proxy target: ${proxyTarget}`);
});
