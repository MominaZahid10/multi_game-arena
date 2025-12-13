import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig(async ({ mode }) => {
  // Only enable the optional development component tagger when explicitly requested.
  // The `lovable-tagger` plugin has caused unexpected runtime instrumentation in some
  // development environments; keep it opt-in via ENABLE_TAGGER=1 to avoid hard-to-debug
  // issues during local dev. This allows you to opt into the plugin when needed.
  let taggerPlugin: any = null;
  if (mode === "development" && process.env.ENABLE_TAGGER === '1') {
    try {
      const mod: any = await import("lovable-tagger");
      if (mod && typeof mod.componentTagger === "function") {
        taggerPlugin = mod.componentTagger();
      }
    } catch (err) {
      // Fail silently but log to console for debugging
      // eslint-disable-next-line no-console
      console.warn('lovable-tagger plugin failed to load:', err);
      taggerPlugin = null;
    }
  }

  return {
    server: {
      host: true,
      port: 8080,
      proxy: {
        '/api': {
          target: 'http://localhost:8000',
          changeOrigin: true,
          ws: true,
          timeout: 30000,  // 30 second timeout
          proxyTimeout: 30000,  // 30 second proxy timeout
          configure: (proxy, _options) => {
            proxy.on('error', (err, _req, _res) => {
              console.log('❌ Proxy error:', err);
            });
            proxy.on('proxyReq', (proxyReq, req, _res) => {
              console.log('→ Sending request to target:', req.method, req.url);
            });
            proxy.on('proxyRes', (proxyRes, req, _res) => {
              console.log('← Received response from target:', req.method, req.url, 'Status:', proxyRes.statusCode);
            });
          },
        },
      },
    },
    plugins: [react(), taggerPlugin].filter(Boolean) as any,
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
  };
});
