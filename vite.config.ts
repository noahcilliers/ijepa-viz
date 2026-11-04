import { createReadStream, statSync } from 'node:fs';
import { extname, join, normalize } from 'node:path';
import { defineConfig, type Plugin } from 'vite';

const MIME: Record<string, string> = {
  '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp',
  '.json': 'application/json', '.bin': 'application/octet-stream', '.npy': 'application/octet-stream',
};

/**
 * Serve files under public/ straight from disk. Vite snapshots the public folder at startup,
 * so files that serve.py writes while the page is open would otherwise fall through to the
 * SPA index.html.
 */
function livePublic(): Plugin {
  return {
    name: 'live-public',
    configureServer(server) {
      const root = server.config.publicDir;
      server.middlewares.use((req, res, next) => {
        const url = (req.url ?? '').split('?')[0];
        const ext = extname(url).toLowerCase();
        if (!(ext in MIME) || url.includes('..')) return next();
        const file = normalize(join(root, decodeURIComponent(url)));
        if (!file.startsWith(root)) return next();
        let st;
        try {
          st = statSync(file);
        } catch {
          return next();
        }
        if (!st.isFile()) return next();
        res.setHeader('Content-Type', MIME[ext]);
        res.setHeader('Content-Length', String(st.size));
        res.setHeader('Cache-Control', 'no-cache');
        createReadStream(file).pipe(res);
      });
    },
  };
}

export default defineConfig({
  plugins: [livePublic()],
  server: {
    // the embedding server (serve.py) answers /api; keep the page on one origin
    proxy: {
      '/api': { target: 'http://127.0.0.1:8765', changeOrigin: true },
    },
    // serve.py writes into public/data while the page is open; don't reload the page for that
    watch: { ignored: ['**/public/**'] },
  },
});
