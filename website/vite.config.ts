import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import {
  cpSync,
  mkdirSync,
  createReadStream,
  existsSync,
  readdirSync,
  statSync,
} from "node:fs";
import { resolve, sep, extname } from "node:path";
const resourcePaths = [
  "notes",
  "materials",
  ...readdirSync("..").filter((name) => name.endsWith(".md")),
  "source_manifest.tsv",
  "checksums.sha256",
  "website/README.md",
  "website/DELIVERY.md",
  "website/VERIFICATION.md",
  "website/ARCHITECTURE.md",
];
export default defineConfig({
  base: process.env.BASE_PATH || "./",
  plugins: [
    react(),
    {
      name: "course-resources",
      configureServer(server) {
        const root = resolve("..");
        server.middlewares.use("/resources/", (req, res, next) => {
          let path: string;
          try {
            path = resolve(
              root,
              decodeURIComponent(
                (req.url || "").split("?")[0].replace(/^\//, ""),
              ),
            );
          } catch {
            res.statusCode = 400;
            res.end();
            return;
          }
          const allowed = resourcePaths.some(
            (p) =>
              path === resolve(root, p) ||
              path.startsWith(resolve(root, p) + sep),
          );
          if (
            !allowed ||
            !path.startsWith(root + sep) ||
            !existsSync(path) ||
            !statSync(path).isFile()
          ) {
            next();
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
        mkdirSync("dist/resources", { recursive: true });
        for (const path of resourcePaths) {
          mkdirSync(resolve("dist/resources", path, ".."), { recursive: true });
          cpSync("../" + path, "dist/resources/" + path, { recursive: true });
        }
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
