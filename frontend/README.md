# Seen'emAll frontend

The React frontend uses Vite for development, production builds, and Vitest for tests.

## Commands

- `npm start` starts the development server at <http://localhost:5173>.
- `npm test` runs the frontend test suite once.
- `npm run build` type-checks the application and writes a production bundle to `dist/`.

The development server proxies API routes to `API_PROXY_TARGET`, then `VITE_API_URL`, and finally `http://localhost:8000`. Set `API_AUTH_KEY` for the proxy to attach an API key without exposing it in the browser bundle.

`VITE_API_URL` and `VITE_API_KEY` are compiled into the browser bundle when present. Use them only when the browser must call the API directly; do not place secrets in `VITE_*` variables.

The production image builds the Vite bundle and serves it through `server.js` on port 3000.
