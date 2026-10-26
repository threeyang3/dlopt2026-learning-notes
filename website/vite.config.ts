import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { createReadStream } from "node:fs";
import { resolve, extname } from "node:path";
import { resourcePath, copyResources } from "./scripts/resources";
export default defineConfig({
  appType: "mpa",
  base: process.env.BASE_PATH || "./",
  plugins: [
    react(),
    {
      name: "course-resources",
      configureServer(server) {
        const root = resolve("..");
        server.middlewares.use("/resources/", (req, res, next) => {
          const path = resourcePath(root, req.url || "");
          if (!path) {
            res.statusCode = 404;
            res.end("Resource not published");
            return;
          }
          const mime: Record<string, string> = {
            ".pdf": "application/pdf",
            ".md": "text/plain; charset=utf-8",
          };
          res.setHeader(
            "Content-Type",
            mime[extname(path)] || "application/octet-stream",
          );
          createReadStream(path)
            .on("error", () => {
              res.statusCode = 404;
              res.end();
            })
            .pipe(res);
        });
      },
      closeBundle() {
        copyResources(resolve(".."), resolve("dist/resources"));
      },
    },
  ],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("node_modules/katex")) return "math-renderer";
          if (id.includes("node_modules/ml-matrix")) return "matrix-engine";
          if (
            id.includes("node_modules") &&
            /remark|rehype|micromark|mdast|hast|unified|react-markdown/.test(id)
          )
            return "note-reader";
        },
      },
    },
  },
});
