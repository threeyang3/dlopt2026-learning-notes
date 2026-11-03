import React from "react";
import { createRoot } from "react-dom/client";
import "katex/dist/katex.min.css";
import { App } from "./App";
import "./styles/app.css";
import "./styles/note-layout.css";
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
