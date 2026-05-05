import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import dotenv from "dotenv";

dotenv.config();

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // API Proxy to Backend (:5000)
  app.use('/api', (req, res, next) => {
    req.url = '/api' + req.url;
    req.headers.host = 'localhost:5000';
    fetch(`http://localhost:5000${req.url}`, {
      method: req.method,
      headers: {
        'Content-Type': 'application/json',
        ...req.headers
      },
      body: req.method !== 'GET' ? JSON.stringify(req.body) : undefined
    }).then(r => r.json()).then(data => res.json(data)).catch(next);
  });

  // Health check (direct)
  app.get("/api/health", (req, res) => {
    res.json({ status: "ok", system: "FactoryGuard AI Production Node", backend_proxy: true });
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`FactoryGuard AI Server running at http://localhost:${PORT}`);
  });
}

startServer();
