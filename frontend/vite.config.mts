import react from '@vitejs/plugin-react';
import { loadEnv } from 'vite';
import { defineConfig } from 'vitest/config';

const proxyPaths = ['/recommend', '/user', '/availability', '/health'];

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const target = env.API_PROXY_TARGET || env.VITE_API_URL || 'http://localhost:8000';
  const apiKey = env.API_AUTH_KEY || env.VITE_API_KEY || '';
  const headers = apiKey ? { 'X-API-Key': apiKey } : undefined;

  return {
    plugins: [react()],
    server: {
      proxy: Object.fromEntries(
        proxyPaths.map((path) => [path, { target, changeOrigin: true, headers }])
      ),
    },
    test: {
      environment: 'jsdom',
      setupFiles: './src/setupTests.ts',
    },
  };
});
