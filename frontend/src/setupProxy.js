const { createProxyMiddleware } = require('http-proxy-middleware');

const defaultTarget = 'http://localhost:8000';
const target =
  process.env.API_PROXY_TARGET ||
  process.env.REACT_APP_API_URL ||
  defaultTarget;

const proxyPaths = [
  '/recommend',
  '/user',
  '/availability',
  '/health',
];

module.exports = function configureProxy(app) {
  proxyPaths.forEach((path) => {
    app.use(
      path,
      createProxyMiddleware({
        target,
        changeOrigin: true,
        logLevel: 'warn',
      })
    );
  });
};
