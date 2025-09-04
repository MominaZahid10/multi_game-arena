import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig(async ({ mode }) => {
  let taggerPlugin: any = null;
  if (mode === "development") {
    try {
      const mod: any = await import("lovable-tagger");
      if (mod && typeof mod.componentTagger === "function") {
        taggerPlugin = mod.componentTagger();
      }
    } catch {
      taggerPlugin = null;
    }
  }

  return {
    server: {
      host: "::",
      port: 8080,
    },
    plugins: [react(), taggerPlugin].filter(Boolean) as any,
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
  };
});
